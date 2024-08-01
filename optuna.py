import csv
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import time
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import ast
import math
import optuna
import argparse
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
import numpy as np
from sklearn.cluster import DBSCAN
import sys
sys.setrecursionlimit(10**6)
parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help'
)

parser.add_argument("--priorityEfficencyaGoodTrack", "-prEfftrk", help="Priority for Efficiency good track", type=int, default=100)
parser.add_argument("--priorityEfficencyaParticleReco", "-prEffreco", help="Priority for Efficiency particle reco", type=int, default=100)
parser.add_argument("--priorityDuplicateRate", "-prdup", help="Priority for duplicate rate", type=int, default=100)
parser.add_argument("--priorityFackRate", "-prfack", help="Priority for Efficiency Fack rate", type=int, default=100)
parser.add_argument("--priorityInferenceTime", "-prtime", help="Priority for Inference time ", type=int, default=100)

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def computeLoss(
    score_good: torch.Tensor,
    score_duplicate: list[torch.Tensor],
    batch_loss: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Compute one loss for each duplicate track associated with the particle"""
    if score_duplicate:
        for s in score_duplicate:
            batch_loss += F.relu(s - score_good + margin) / len(score_duplicate)
    return batch_loss

class DuplicateClassifier(nn.Module):
    """MLP model used to separate good tracks from duplicate tracks. Return one score per track the higher one correspond to the good track."""

    def __init__(self, input_dim, n_layers):
        super(DuplicateClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for n_units in n_layers:
            layers.append(nn.Linear(prev_dim, n_units))
            layers.append(nn.ReLU())  # Adding ReLU activation function
            prev_dim = n_units
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Final activation function
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)

class Normalise(nn.Module):
    """Normalization of the input before the MLP model."""

    def __init__(self, mean, std):
        super(Normalise, self).__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def forward(self, z):
        z = z - self.mean
        z = z / self.std
        return z

def batchSplit(data: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    """Split the data into batch each containing @batch_size truth particles (the number of corresponding tracks may vary)"""
    batch = []
    pid = data[0][0]
    n_particle = 0
    id_prev = 0
    id = 0
    for index, row, truth in zip(data[0], data[1], data[2]):
        if index != pid:
            pid = index
            n_particle += 1
            if n_particle == batch_size:
                b = data[0][id_prev:id], data[1][id_prev:id], data[2][id_prev:id]
                batch.append(b)
                n_particle = 0
                id_prev = id
        id += 1
    return batch

def renameCluster(clusterarray: np.ndarray) -> np.ndarray:
    """Rename the cluster IDs to be int starting from 0"""
    last_id = -1
    new_id = -1
    for track in clusterarray:
        if track[1] != last_id:
            last_id = track[1]
            new_id = new_id + 1
        track[1] = new_id
    return clusterarray

def subClustering(clusterarray: np.ndarray, c: int, lastCluster: float) -> np.ndarray:
    """SubClustering algorithm, cluster together tracks that share hits (TODO : doesn't handle real shared hits)"""
    newCluster = math.nextafter(lastCluster, c + 1)
    if newCluster >= c + 1:
        raise RuntimeError(
            "Too many subcluster in the clusters, this shouldn't be possible."
        )
    hits_IDs = []
    set_IDs = set(hits_IDs)
    for track in clusterarray:
        if track[1] == c:
            if hits_IDs == []:
                hits_IDs = track[0]
                set_IDs = set(hits_IDs)
            if set_IDs & set(track[0]):
                track[1] = newCluster
    if hits_IDs == []:
        return clusterarray
    else:
        clusterarray = subClustering(clusterarray, c, newCluster)
        return clusterarray

def scoringBatch(batch: list[pd.DataFrame], duplicateClassifier, device, margin, Optimiser=0) -> tuple[int, int, float]:
    """Run the MLP on a batch and compute the corresponding efficiency and loss. If an optimiser is specified train the MLP."""
    nb_part = 0
    nb_good_match = 0
    loss = 0
    max_score = 0
    max_match = 0
    for b_data in batch:
        pid = b_data[0][0]
        batch_loss = 0
        score_good = 0
        score_duplicate = []
        if Optimiser:
            Optimiser.zero_grad()
        input = torch.tensor(b_data[1], dtype=torch.float32).to(device)
        prediction = duplicateClassifier(input)
        for index, pred, truth in zip(b_data[0], prediction, b_data[2]):
            if index != pid:
                if max_match == 1:
                    nb_good_match += 1
                batch_loss = computeLoss(score_good, score_duplicate, batch_loss, margin)
                nb_part += 1
                pid = index
                score_duplicate = []
                score_good = 0
                max_score = 0
                max_match = 0
            if truth:
                score_good = pred
            else:
                score_duplicate.append(pred)
            if pred == max_score:
                max_match = 0
            if pred > max_score:
                max_score = pred
                max_match = truth
        if max_match == 1:
            nb_good_match += 1
        batch_loss = computeLoss(score_good, score_duplicate, batch_loss, margin)
        nb_part += 1
        batch_loss = batch_loss / len(b_data[0])
        loss += batch_loss.item()
        if Optimiser:
            batch_loss.backward()
            Optimiser.step()
    loss = loss / len(batch)
    return nb_part, nb_good_match, loss

def readDataSet(CKS_files: list[str]) -> pd.DataFrame:
    data = []
    for f in CKS_files:
        datafile = pd.read_csv(f)
        datafile = prepareDataSet(datafile)
        data.append(datafile)
    return data

def clusterTracks(event: pd.DataFrame, DBSCAN_eps: float = 0.07, DBSCAN_min_samples: int = 2) -> pd.DataFrame:
    trackDir = event[["eta", "phi"]].to_numpy()
    clustering = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples).fit(trackDir)
    event["cluster"] = clustering.labels_
    sorted = event.sort_values(["cluster", "nMeasurements"], ascending=[True, False])
    updatedCluster = []
    cluster_hits = sorted.loc[:, ("Hits_ID", "cluster")]
    for key, frame in cluster_hits.groupby("cluster"):
        clusterarray = frame.to_numpy()
        clusterarray = subClustering(clusterarray, key, key)
        updatedCluster.extend(clusterarray[:, 1])
    sorted.loc[:, ("cluster")] = updatedCluster
    sorted = sorted.sort_values("cluster")
    clusterarray = sorted.loc[:, ("Hits_ID", "cluster")].to_numpy()
    clusterarray = renameCluster(clusterarray)
    sorted.loc[:, ("cluster")] = clusterarray[:, 1]
    return sorted

def prepareInferenceData(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    target_column = "good/duplicate/fake"
    y = LabelEncoder().fit(data[target_column]).transform(data[target_column])
    input = data.drop(
        columns=[
            target_column,
            "track_id",
            "nMajorityHits",
            "nSharedHits",
            "truthMatchProbability",
            "Hits_ID",
            "chi2",
            "pT",
            "cluster","seed_id"
        ]
    )
    x_cat = OrdinalEncoder().fit_transform(input.select_dtypes("object"))
    x = np.concatenate((x_cat, input), axis=1)
    return x, y

def prepareTestingData(data: pd.DataFrame, avg_mean, avg_sdv, events) -> tuple[np.ndarray, np.ndarray, list, list, int]:
    target_column = "good/duplicate/fake"
    y = LabelEncoder().fit(data[target_column]).transform(data[target_column])
    input = data.drop(
        columns=[
            target_column,
            "track_id",
            "nMajorityHits",
            "nSharedHits",
            "truthMatchProbability",
            "Hits_ID",
            "chi2",
            "pT","seed_id"
        ]
    )
    scale = StandardScaler()
    scale.fit(input.select_dtypes("number"))
    avg_mean = avg_mean + scale.mean_
    avg_sdv = avg_sdv + scale.var_
    events += 1
    x_cat = OrdinalEncoder().fit_transform(input.select_dtypes("object"))
    x = np.concatenate((x_cat, input), axis=1)
    return x, y, avg_mean, avg_sdv, events

def prepareDataSet(data: pd.DataFrame) -> pd.DataFrame:
    data = data[data["nMeasurements"] > 6]
    data = data.drop_duplicates(
        subset=[
            "particleId",
            "Hits_ID",
            "nOutliers",
            "nHoles",
            "nSharedHits",
            "chi2",
        ],
        keep="first",
    )
    data = data.sort_values("particleId")
    data = data.set_index("particleId")
    hitsIds = []
    for list in data["Hits_ID"].values:
        hitsIds.append(ast.literal_eval(list))
    data["Hits_ID"] = hitsIds
    return data

def train(
    duplicateClassifier: DuplicateClassifier,
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
    device,
    margin,
    epochs: int = 20,
    batch: int = 32,
    validation: float = 0.3,
) -> DuplicateClassifier:
    writer = SummaryWriter()
    opt = torch.optim.Adam(duplicateClassifier.parameters())
    batch = batchSplit(data, batch)
    val_batch = int(len(batch) * (1 - validation))
    for epoch in range(epochs):
        print("Epoch: ", epoch, " / ", epochs)
        loss = 0.0
        nb_part = 0.0
        nb_good_match = 0.0
        nb_part, nb_good_match, loss = scoringBatch(batch[:val_batch], duplicateClassifier, device, margin, Optimiser=opt)
        print("Loss/train: ", loss, " Eff/train: ", nb_good_match / nb_part)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Eff/train", nb_good_match / nb_part, epoch)
        if validation > 0.0:
            nb_part, nb_good_match, loss = scoringBatch(batch[val_batch:], duplicateClassifier, device, margin)
            writer.add_scalar("Loss/val", loss, epoch)
            writer.add_scalar("Eff/val", nb_good_match / nb_part, epoch)
            print("Loss/val: ", loss, " Eff/val: ", nb_good_match / nb_part)
    writer.close()
    return duplicateClassifier

def readDataSett(CKS_files : list[str]) -> pd.DataFrame:
    """Read the dataset from the different files, remove the pure duplicate tracks and combine the datasets"""
    data = pd.DataFrame()
    for f in CKS_files:
        datafile = pd.read_csv(f)
        # We at this point we don't make any difference between fake and duplicate
        datafile.loc[
            datafile["good/duplicate/fake"] == "fake", "good/duplicate/fake"
        ] = "duplicate"
        datafile = prepareDataSet(datafile)
        # Combine dataset
        data = pd.concat([data, datafile])
    return data

def objective(trial):
    """
    The objective function to be optimized by Optuna.

    Args:
    trial (optuna.trial.Trial): A trial object to suggest hyperparameters.

    Returns:
    float: The objective value to minimize or maximize.
    """
    # Suggest the number of layers and units in each layer for the MLP
    n_layers = trial.suggest_int('n_layers', 1, 5)
    layers = [trial.suggest_int(f'n_units_{i}', 4, 128) for i in range(n_layers)]
    
    # Suggest the margin value for the loss computation
    margin = trial.suggest_uniform('margin', 0.001, 0.20)
    
    # Define file path pattern and read dataset files
    CKF_files = sorted(glob.glob("/data/atlas/callaire/Acts/ODD_data/odd_full_chain_01" + "/event0000000[0-9][0-9]-tracks_ckf.csv"))
    data = readDataSet(CKF_files)
    
    clusteredData = []
    cleanedData = []

    # Cluster tracks for each event
    for event in data:
        clustered = clusterTracks(event)
        clusteredData.append(clustered)

    # Define another file path pattern and read additional dataset files
    CKF_filles = sorted(glob.glob("/data/atlas/callaire/Acts/ODD_data/odd_full_chain_02" + "/event0000000[0-9][0-9]-tracks_ckf.csv"))
    dataa = readDataSett(CKF_filles)

    avg_mean = [0, 0, 0, 0, 0, 0, 0, 0]
    avg_sdv = [0, 0, 0, 0, 0, 0, 0, 0]
    events = 0

    # Prepare the testing data
    x_train, y_train, avg_mean, avg_sdv, events = prepareTestingData(dataa, avg_mean, avg_sdv, events)

    avg_mean = [x / events for x in avg_mean]
    avg_sdv = [x / events for x in avg_sdv]
    input_dim = np.shape(x_train)[1]

    # Initialize the DuplicateClassifier model with normalization
    duplicateClassifier = nn.Sequential(
        Normalise(avg_mean, avg_sdv), 
        DuplicateClassifier(input_dim, layers)
    )
    duplicateClassifier = duplicateClassifier.to(device)
    
    # Prepare input data for training
    input = dataa.index, x_train, y_train
    
    # Train the DuplicateClassifier model
    train(duplicateClassifier, input, device, margin, epochs=20, batch=128, validation=0.3)
    duplicateClassifier.eval()

    # Suggest an optimizer and learning rate
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    optimizer = getattr(torch.optim, optimizer_name)(duplicateClassifier.parameters(), lr=lr)

    # Measure the inference time
    timeStartInference = time.time()
    for clusteredEvent in clusteredData:
        x_test, y_test = prepareInferenceData(clusteredEvent)
        output_predict = []
        for x in x_test:
            x = torch.tensor(x, dtype=torch.float32).to(device)
            output_predict.append(duplicateClassifier(x).item())

        clusteredEvent["score"] = output_predict
        cleanedEvent = clusteredEvent
        
        # Keep only the best scored track per cluster
        idx = (
            cleanedEvent.groupby(["cluster"])["score"].transform(max)
            == cleanedEvent["score"]
        )
        cleanedEvent = cleanedEvent[idx]
        cleanedData.append(cleanedEvent)

    timeEndInference = time.time()

    # Initialize counters for various metrics
    nb_part = 0
    nb_track = 0
    nb_fake = 0
    nb_duplicate = 0

    nb_good_match = 0
    nb_reco_part = 0
    nb_reco_fake = 0
    nb_reco_duplicate = 0
    nb_reco_track = 0

    # Calculate metrics for each event
    for clusteredEvent, cleanedEvent in zip(clusteredData, cleanedData):
        nb_part += clusteredEvent.loc[
            clusteredEvent["good/duplicate/fake"] != "fake"
        ].index.nunique()
        nb_track += clusteredEvent.shape[0]
        nb_fake += clusteredEvent.loc[
            clusteredEvent["good/duplicate/fake"] == "fake"
        ].shape[0]
        nb_duplicate += clusteredEvent.loc[
            clusteredEvent["good/duplicate/fake"] == "duplicate"
        ].shape[0]
        nb_good_match += cleanedEvent.loc[
            cleanedEvent["good/duplicate/fake"] == "good"
        ].shape[0]
        nb_reco_fake += cleanedEvent.loc[
            cleanedEvent["good/duplicate/fake"] == "fake"
        ].shape[0]
        nb_reco_duplicate += cleanedEvent.loc[
            cleanedEvent["good/duplicate/fake"] == "duplicate"
        ].shape[0]
        nb_reco_part += cleanedEvent.loc[
            cleanedEvent["good/duplicate/fake"] != "fake"
        ].index.nunique()
        nb_reco_track += cleanedEvent.shape[0]

    # Calculate efficiency and other rates
    efficiency_good_track = 100 * nb_good_match / nb_part
    efficiency_particle_reco = 100 * nb_reco_part / nb_part
    duplicate_rate = 100 * ((nb_good_match + nb_reco_duplicate) - nb_reco_part) / nb_reco_track
    fake_rate = 100 * nb_reco_fake / nb_reco_track
    inference_time = (timeEndInference - timeStartInference) * 1000 / len(CKF_files)

    # Print calculated metrics
    print("Efficiency (good track) : ", efficiency_good_track, " %")
    print("Efficiency (particle reco) : ", efficiency_particle_reco, " %")
    print("Duplicate rate: ", duplicate_rate, " %")
    print("Fake rate: ", fake_rate, " %")
    print("Inference : ", inference_time, "ms")
    print("Priority : ", args.priorityEfficencyaGoodTrack, args.priorityEfficencyaParticleReco, args.priorityDuplicateRate, args.priorityFackRate, args.priorityInferenceTime)
    print("First  : ", (nb_good_match / nb_part)*args.priorityEfficencyaGoodTrack)
    print("Second :", (nb_reco_part / nb_part) * args.priorityEfficencyaParticleReco)
    print("Third :", (1- ((nb_good_match + nb_reco_duplicate) - nb_reco_part) / nb_reco_track) * args.priorityDuplicateRate)
    print("Fourth : ", (1- nb_reco_fake / nb_reco_track) * args.priorityFackRate)
    print("Fifth: ", ((timeEndInference-timeStartInference) / len(CKF_files))*args.priorityInferenceTime)

    # Write results to CSV file in append mode
    csv_file = 'optuna_result_per_batch.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if the file does not exist
            writer.writerow([
                'Optimizer', 'Number of Layers', 'Learning Rate', 
                *[f'Units in Layer {i}' for i in range(n_layers)],
                'Efficiency (good track) %', 'Efficiency (particle reco) %', 
                'Duplicate Rate %', 'Fake Rate %', 'Inference Time ms', 'Margin'
            ])
        # Write data row
        writer.writerow([
            optimizer_name, n_layers, lr, 
            *layers, 
            efficiency_good_track, efficiency_particle_reco, 
            duplicate_rate, fake_rate, inference_time, margin
        ])

    # Calculate the objective value based on the priorities
    return (nb_good_match / nb_part)*args.priorityEfficencyaGoodTrack + (nb_reco_part / nb_part) * args.priorityEfficencyaParticleReco + (1- ((nb_good_match + nb_reco_duplicate) - nb_reco_part) / nb_reco_track) * args.priorityDuplicateRate + (1- nb_reco_fake / nb_reco_track) * args.priorityFackRate - ((timeEndInference-timeStartInference) / len(CKF_files))*args.priorityInferenceTime

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, n_jobs=3)
    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
