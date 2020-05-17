from collections import defaultdict, Counter
import csv
import random
import json
from nltk.metrics import agreement

# constants
HITID = "HITId"
LABEL = "Answer.political bias.label"
WORKERID = "WorkerId"
WORKTIME = "WorkTimeInSeconds"
APPROVE = "Approve"
TEXT = "Input.text"
sample_path = "amt_output_csv/abortion_batch_results.csv"


class CorpusAnalysis(object):
    def __init__(self, data_path):
        self.table_titiles = list()
        # ['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward', 'CreationTime', 'MaxAssignments', 'RequesterAnnotation', 'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds', 'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds', 'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime', 'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime', 'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate', 'Last30DaysApprovalRate', 'Last7DaysApprovalRate', 'Input.video_title', 'Input.policy', 'Input.media', 'Input.text', 'Answer.political bias.label', 'Approve', 'Reject']
        self.full_table = list()
        self.hitid_labels = defaultdict(list)
        self.hitid_goldlabel = defaultdict(str)
        self.hitid_majoritylabel = defaultdict(str)
        self.hit_adjudicate = defaultdict(list)
        with open(data_path, mode="r") as infile:
            reader = csv.reader(infile)
            for i, row in enumerate(reader):
                if i == 0:
                    self.table_titiles = row
                else:
                    self.full_table.append(row)
                    self.hitid_labels[row[0]].append(row[-1])
        self.title_index = {k: v for v, k in enumerate(self.table_titiles)}
        self.policy = self.full_table[0][self.title_index["Input.policy"]]

    def populate_hitid_goldlabel(self):
        # get the majority voting as the gold label in hit_goldlabel
        # me as adjudicator breaking ties manually in hit_adjudicate
        for k, v in self.hitid_labels.items():
            majority_label = Counter(v).most_common()[0][0]
            majority_label_count = Counter(v).most_common()[0][1]
            if len(v) == 3 and majority_label_count != 1:
                self.hitid_goldlabel[k] = majority_label
                self.hitid_majoritylabel[k] = majority_label
            else:
                self.hit_adjudicate[k] = v
        # get Majority aggregation/ties
        # print(len(self.hit_goldlabel))
        # print(len(self.hit_adjudicate.keys()))

        ##TODO:change this when get full data and manually adjudicated
        for k, v in self.hit_adjudicate.items():
            self.hitid_goldlabel[k] = v[0]

        # adjudicate, get the gold labels
        for row in self.full_table:
            hit_id = row[self.title_index[HITID]]
            label = row[self.title_index[LABEL]]
            if label == self.hitid_goldlabel.get(hit_id, "non-exist"):
                row.append("Approved")
            else:
                row.append("Rejected")

        # get label distribution:
        # print(Counter(self.hit_goldlabel.values()))
        # print("*****************************************")

    def turker_accuracy(self):
        # get how many turkers got it right/wrong
        adjudication_list = list()
        for row in self.full_table:
            adjudication_list.append(row[-1])
        # print("*****************************************")
        # print(Counter(adjudication_list))

        worker_app_rej = defaultdict(list)
        for row in self.full_table:
            if row[self.title_index[APPROVE]] == "Approved":
                if worker_app_rej[row[self.title_index[WORKERID]]]:
                    worker_app_rej[row[self.title_index[WORKERID]]][0] += 1
                else:
                    worker_app_rej[row[self.title_index[WORKERID]]].append(1)
                    worker_app_rej[row[self.title_index[WORKERID]]].append(0)
            else:
                if worker_app_rej[row[self.title_index[WORKERID]]]:
                    worker_app_rej[row[self.title_index[WORKERID]]][1] += 1
                else:
                    worker_app_rej[row[self.title_index[WORKERID]]].append(0)
                    worker_app_rej[row[self.title_index[WORKERID]]].append(1)

        worker_error_rate = {
            k: [v[0] / (v[0] + v[1]), v[0] + v[1]] for k, v in worker_app_rej.items()
        }
        sorted_worker_error_rate = {
            k: v
            for k, v in sorted(
                worker_error_rate.items(), key=lambda item: item[1][1], reverse=True
            )
        }
        with open("turker_accuracy/{}.json".format(self.policy), "w") as f:
            json.dump(sorted_worker_error_rate, f, indent=2)
        x = sum(a[0] for a in sorted_worker_error_rate.values())
        y = sum(a[1] for a in sorted_worker_error_rate.values())
        length = len(sorted_worker_error_rate)

        return x / length, y / length

    # def get_iaa(self):
    # iaa_data = list()
    # prev_hitid = full_table[0][title_index[HITID]]
    # for i in range(0, len(full_table), 3):
    #     iaa_data.append([0, full_table[i][title_index[HITID]], full_table[i][title_index[LABEL]]])
    #     iaa_data.append([1, full_table[i+1][title_index[HITID]], full_table[i+1][title_index[LABEL]]])
    #     iaa_data.append([2, full_table[i+2][title_index[HITID]], full_table[i+2][title_index[LABEL]]])
    #
    # task = agreement.AnnotationTask(data=iaa_data)
    # print(task.kappa())
    # print(task.alpha())
    def get_data(self):

        self.hitid_text = defaultdict(str)
        for row in self.full_table:
            self.hitid_text[row[self.title_index[HITID]]] = row[self.title_index[TEXT]]

        text_adjudicate = set()
        for id in self.hit_adjudicate:
            text_adjudicate.add(self.hitid_text[id])
        # print(text_adjudicate)
        # with open('tied_sents/{}.txt'.format(self.policy), 'w') as f:
        #     f.write("\n\n".join(text_adjudicate))

    def get_training_data(self):
        data = [["text", "label"]]
        for id, label in self.hitid_goldlabel.items():
            data.append([self.hitid_text[id], label])
        with open("unsplitted_data/{}.csv".format(self.policy), "w") as out:
            csv_out = csv.writer(out)
            for row in data:
                csv_out.writerow(row)

    def get_avg_accuracy(self):
        agreed = 0
        disagreed = 0
        for id, labels_list in self.hitid_labels.items():
            for label in labels_list:
                if label == self.hitid_goldlabel[id]:
                    agreed += 1
                else:
                    disagreed += 1
        return agreed / (agreed + disagreed)

    def get_wawa(self):
        agreed = 0
        disagreed = 0
        for id, labels_list in self.hitid_labels.items():
            for label in labels_list:
                if label == self.hitid_majoritylabel[id]:
                    agreed += 1
                else:
                    disagreed += 1
        return agreed / (agreed + disagreed)

    def get_random_sampling_accuracy(self, num_sample=100):
        keys = random.sample(self.hitid_labels.keys(), num_sample)
        agreed = 0
        disagreed = 0
        for id in keys:
            for label in self.hitid_labels[id]:
                if label == self.hitid_goldlabel[id]:
                    agreed += 1
                else:
                    disagreed += 1
        return agreed / (agreed + disagreed)


if __name__ == "__main__":
    policies = [
        "healthcare",
        "economic",
        "immigration",
        "education",
        "abortion",
        "LGBTQ",
        "gun",
        "environment",
    ]
    data_paths = [
        "amt_output_csv/{}_batch_results.csv".format(policy) for policy in policies
    ]

    for path in data_paths:
        ca = CorpusAnalysis(path)
        ca.populate_hitid_goldlabel()
        print(ca.turker_accuracy())
        ca.get_data()
        ca.get_training_data()
    #     print("*******************************")
    #     print(ca.get_avg_accuracy())
    #     print(ca.get_wawa())
    #     print(ca.get_random_sampling_accuracy())

    # path = '/Users/jinzhao/Desktop/4th_semester/thesis/thesis/amt_output_csv/healthcare_batch_results.csv'
    # ca = CorpusAnalysis(path)
    # ca.populate_hitid_goldlabel()
    # ca.turker_accuracy()
    # ca.get_data()
    # ca.get_training_data()
