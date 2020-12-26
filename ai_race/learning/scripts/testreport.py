import json


class TestReport:
    def __init__(self):
        self.report = []

    # add new classification report into self.report
    def append(self, epoch_n, class_repo, conf_mat):
        class_repo_split=class_repo.splitlines()
        for i in range(len(class_repo_split)):
            if "precision" in class_repo_split[i]:
                startline = i
            elif "accuracy" in class_repo_split[i]:
                accline = i
                break

        report1 = []
        for j in range(startline+2,accline-1):
            class_repo_spsp = class_repo_split[j].split(' ')
            class_repo_spsp_ext = []
            for content in class_repo_spsp:
                if content != "":
                    class_repo_spsp_ext.append(content)
            report1.append(class_repo_spsp_ext)

        report2 = []
        for j in range(accline, accline+3):
            class_repo_spsp = class_repo_split[j].split(' ')
            class_repo_spsp_ext = []
            for content in class_repo_spsp:
                if content != "":
                    class_repo_spsp_ext.append(content)
            report2.append(class_repo_spsp_ext)

        report1json={}
        dat = {"epoch":epoch_n}
        report1json.update(dat)
        
        for j in range(len(report1)):
            dat = {report1[j][0]:{"precision":report1[j][1]
                            ,"recall":report1[j][2]
                            ,"f1-score":report1[j][3]
                            ,"support":report1[j][4]
                            }
                }
            report1json.update(dat)

        for j in range(len(report2)):
            if j == 0:
                dat = {report2[j][0]:{"precision":""
                                ,"recall":""
                                ,"f1-score":report2[j][1]
                                ,"support":report2[j][2]
                                }
                    }
            else:
                dat = {report2[j][0]+" "+report2[j][1]:{"precision":report2[j][2]
                                ,"recall":report2[j][3]
                                ,"f1-score":report2[j][4]
                                ,"support":report2[j][5]
                                }
                    }
            report1json.update(dat)
        dat = {"confusion_matrix":conf_mat}
        report1json.update(dat)

        self.report.append(report1json)

    def dump(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.report, f, indent=4)


if __name__ == "__main__":
    trepo = TestReport()


    clas = """/usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:1221: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
    _warn_prf(average, modifier, msg_start, len(result))
                precision    recall  f1-score   support

            0       0.00      0.00      0.00         0
            1       0.75      0.81      0.78       138
            2       0.88      0.83      0.85       222

        accuracy                        0.82       360
    macro avg       0.54      0.55      0.54       360
    weighted avg    0.83      0.82      0.82       360"""

    conf = [[  0 ,  0 ,  0]
        ,[  2 , 51 , 10]
        ,[  0 , 95 ,202]]
    
    trepo.append(3,clas,conf)
    trepo.append(6,clas,conf)
    #trepo.dump("./test.json")
    #print(trepo.report)
    for repo in trepo.report:
        if repo["epoch"] == 6:
            print(repo) 
    #print(json.dumps(trepo.report,indent=4))