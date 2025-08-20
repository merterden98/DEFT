import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, \
    roc_auc_score, accuracy_score, f1_score, average_precision_score
import contextlib
import tempfile
import shlex
import subprocess as sp
from Bio import SeqIO

def run_foldseek_aln(train_folder, test_folder, output_file):

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_hash = hash(test_folder)
        fseek_base_cmd = f"foldseek easy-search --cov-mode 2 -e 0.1 {test_folder} {train_folder} {output_file} {tmpdir}/aln{tmp_hash}"
        print(fseek_base_cmd)
        proc = sp.Popen(
            shlex.split(fseek_base_cmd), stdout=sp.PIPE, stderr=sp.PIPE
        )
        _ = proc.communicate()
        return output_file


def read_aln(path):
    # path contains a tab seperated file with the following columns
    # query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ["Query", "Target", "Fident", "Alnlen", "Mismatch", "Gapopen", "Qstart", "Qend", "Tstart", "Tend", "Evalue", "Bits"]
    # for query and target columns, we only need the protein ID
    df["Query"] = df["Query"].apply(lambda x: x.rstrip(".cif").split("_")[0])
    df["Target"] = df["Target"].apply(lambda x: x.rstrip(".cif").split("_")[0])
    return df


"""
    The Code Below is from CLEAN and used to evaluate the model
    For Purposes of Comparison

"""
def get_eval_metrics(pred_label, true_label, all_label):
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])
    n_test = len(pred_label)
    pred_m = np.zeros((n_test, len(mlb.classes_)))
    true_m = np.zeros((n_test, len(mlb.classes_)))

    for i in range(n_test):
        pred_m[i] = mlb.transform([pred_label[i]])
        true_m[i] = mlb.transform([true_label[i]])
    pre = precision_score(true_m, pred_m, average='weighted', zero_division=0)
    rec = recall_score(true_m, pred_m, average='weighted')
    f1 = f1_score(true_m, pred_m, average='weighted')
    acc = accuracy_score(true_m, pred_m)
    return pre, rec, f1, acc

def restrict_aln(aln, predictions, dataset, db_predictions):
    # Convert predictions to a dictionary where the key is the ID and the value is the prediction

    id_to_pred = dict()
    for k, _, v in predictions:
        id_to_pred[k] = v
    id_to_db_pred = dict()
    for k, _, v in db_predictions:
        id_to_db_pred[k] = v

    # Filter aln if the prediction for the query is the same as the prediction for the target
    aln["Query_Pred"] = aln["Query"].apply(lambda x: id_to_pred[x])
    aln["Target_Pred"] = aln["Target"].apply(lambda x: id_to_db_pred[x])
    aln = aln[aln["Query_Pred"] == aln["Target_Pred"]]
    return aln


def assign_predictions(aln, dataset, predictions, train_csv):
    from collections import defaultdict

    df = pd.read_csv(train_csv)
    id_to_ec = dict()
    frequencies = defaultdict(int)
    for i, row in df.iterrows():
        id_to_ec[row["ID"]] = row["EC"]
        frequencies[row["EC"]] += 1
    aln["EC"] = aln["Target"].apply(lambda x: id_to_ec[x])
    aln["EC_2"] = aln["EC"].apply(lambda x: ".".join(x.split(".")[:2]))

    id_to_ec = dict()
    for i, row in enumerate(dataset):
        id_to_ec[row["ID"]] = predictions[i][2]


    aln["Query_EC2"] = aln["Query"].apply(lambda x: ".".join(id_to_ec[x].split(".")[:2]) if x in id_to_ec else "")
    aln = aln[aln["EC_2"] == aln["Query_EC2"]]
    aln = aln.sort_values(by=["Bits"], ascending=[False]).groupby("Query").head(1000)
    return aln

def add_ec_data(aln, dataset, predictions, train_csv):
    from collections import defaultdict

    df = pd.read_csv(train_csv)
    id_to_ec = defaultdict(str)
    frequencies = defaultdict(int)
    for i, row in df.iterrows():
        id_to_ec[row["ID"]] = row["EC"]
        frequencies[row["EC"]] += 1
    aln["EC"] = aln["Target"].apply(lambda x: id_to_ec[x])
    aln["EC_2"] = aln["EC"].apply(lambda x: ".".join(x.split(".")[:2]))

    id_to_ec = dict()
    id_to_true_ec = dict()

    # The variables, all_labels, true_labels, and pred_labels are used to calculate the scores for CLEAN comparison

    all_labels = set()
    true_labels = []
    pred_labels = []
    for i, row in enumerate(dataset):
        id_to_ec[row["ID"]] = predictions[i][2]
        id_to_true_ec[row["ID"]] = row["EC"]
        for ec in row["EC"].split(";"):
            all_labels.add(ec)
    aln = aln[aln['Query'].isin(id_to_ec.keys())] # Ensure we are only reporting predicted matches
    aln["Query_EC2"] = aln["Query"].apply(lambda x: ".".join(id_to_ec[x].split(".")[:2]))
    aln = aln[aln["EC_2"] == aln["Query_EC2"]]

    aln.to_csv("aln.csv", index=False)
    aln = aln.sort_values(by=["Bits"], ascending=[False]).groupby("Query").head(1000)
    aln["True_EC"] = aln["Query"].apply(lambda x: id_to_true_ec[x])

    
    num_correct = 0
    matches = defaultdict(list)
    true_ec = defaultdict(list)

    for row in aln.itertuples():
        predicted = row.EC.split(";")
        actual = row.True_EC.split(";")
        matches[row.Query] += predicted
        true_ec[row.Query] += actual

    incorrect_matches = []
    duplicates = 0
    num_first_two_correct = 0
    for query in matches:
        true_ecs = set(true_ec[query])
        true_ecs_2 = set([".".join(x.split(".")[:2]) for x in true_ecs])
        found = False
        match_2_found = False
        if len(true_ecs) > 1:
            duplicates += 1
            continue
        preds = []
        for match in matches[query]:
            if match in all_labels:
                preds.append(match)
            match_2 = ".".join(match.split(".")[:2])
            if match_2 in true_ecs_2 and not match_2_found:
                num_first_two_correct += 1
                match_2_found = True
            if match in true_ecs:
                num_correct += 1
                found = True
                break
        pred_labels.append(preds)
        true_labels.append(true_ec[query])
        if not found:
            incorrect_matches += [(query, matches[query], true_ec[query], f"True EC Freq: {frequencies[true_ec[query][0]]}", f"Predicted EC Freq: {[frequencies[x] for x in matches[query]]}")]
    accuracy = num_correct / len(matches)
    accuracy_2 = num_first_two_correct / len(matches)

    
    pre, rec, f1, acc = get_eval_metrics(pred_labels, true_labels, all_labels)

    eval_metrics = {
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "acc": acc
    }
    
    return aln, accuracy, eval_metrics


def extract_3di_from_db(db_path, foldseek="foldseek"):
        CMD = f"{foldseek} convert2fasta {db_path} {db_path}.fasta"
        print(CMD)
        proc = sp.Popen(
            shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE
                    )
        _ = proc.communicate()

        seq_records = SeqIO.to_dict(SeqIO.parse(f"{db_path}.fasta", "fasta"))
        # Update the keys to only have uniprot id 
        seq_records = {key.split("_")[0]: value for key, value in seq_records.items()}
        # create backup of {tmpdir}/{pdb_dir_name}
        CMD = f"cp {db_path} {db_path}_seq"
        proc = sp.Popen(
            shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE
                    )
        _ = proc.communicate()

        # create backup of {tmpdir}/{pdb_dir_name}.index
        CMD = f"cp {db_path}.index {db_path}_seq.index"
        proc = sp.Popen(
            shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE
                    )
        _ = proc.communicate()
        CMD = f"mv {db_path}_ss {db_path}"
        proc = sp.Popen(
            shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE
                    )
        _ = proc.communicate()
        CMD = f"mv {db_path}_ss.index {db_path}.index"
        proc = sp.Popen(
            shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE
                    )
        _ = proc.communicate()
        CMD = f"{foldseek} convert2fasta {db_path} {db_path}_ss.fasta"
        proc = sp.Popen(
            shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE
                    )
        _ = proc.communicate()

        seq_records_struct = SeqIO.to_dict(SeqIO.parse(f"{db_path}_ss.fasta", "fasta"))
        # Update the keys to only have uniprot id 
        seq_records_struct = {key.split("_")[0]: value for key, value in seq_records_struct.items()}


        # Restore the original files
        CMD = f"mv {db_path} {db_path}_ss"
        proc = sp.Popen(
            shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE
                    )
        _ = proc.communicate()


        CMD = f"mv {db_path}.index {db_path}_ss.index"
        proc = sp.Popen(
            shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE
                    )
        _ = proc.communicate()

        CMD = f"mv {db_path}_seq {db_path}"
        proc = sp.Popen(
            shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE
                    )
        _ = proc.communicate()

        CMD = f"mv {db_path}_seq.index {db_path}.index"
        proc = sp.Popen(
            shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE
                    )
        _ = proc.communicate()


        return seq_records, seq_records_struct, db_path


def retrieve_3di(pdb_path, foldseek="foldseek"):
    pdb_dir_name = hash(pdb_path)
    with contextlib.nullcontext(tempfile.mkdtemp()) as tmpdir:
        FSEEK_BASE_CMD = f"{foldseek} createdb {pdb_path} {tmpdir}/{pdb_dir_name}"
        proc = sp.Popen(
            shlex.split(FSEEK_BASE_CMD), stdout=sp.PIPE, stderr=sp.PIPE
        )
        _ = proc.communicate()
        return extract_3di_from_db(f"{tmpdir}/{pdb_dir_name}", foldseek=foldseek)

