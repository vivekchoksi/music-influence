import warnings
from featurizer import FeatureGenerator
import logging
import math
from loader import GraphLoader
import random
import cPickle
import numpy as np
import pdb
import os
from matplotlib import pyplot as plt
from sklearn.metrics.ranking import roc_auc_score, average_precision_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.metrics import precision_score
warnings.filterwarnings("ignore",category=DeprecationWarning)

class EdgePredictor(object):
    # FeatureGenerator from featurizer.py
    featurizer = None

    # Influence graph
    IG = None

    # For now just easy ensemble method
    #classifier = svm.SVC(class_weight="balanced", probability=True)
    classifier = ExtraTreesClassifier()

    # Tuple Xtrain, Phitrain, Ytrain, set when self.train() is called by the client
    train_data = None

    # Tuple Xvalid, Phivalid, Yvalid set when self.train() is called by the client
    validation_data = None

    # Tuple Xtst, Phitst, Ytst set when self.train() is called by the client
    test_data = None


    def __init__(self, IG, verbose=True, features_to_use=None):
        random.seed(0)
        self.basepath = os.path.dirname(os.path.dirname(__file__))
        self.IG = IG
        self.featurizer = FeatureGenerator(verbose=verbose, features_to_use=features_to_use)
        self.verbose = verbose

    def _total_edges(self, IG):
        return len(IG.nodes()) * (len(IG.nodes())-1)  # N * N-1 total possible directed edges

    def all_negative_examples(self, IG):
        examples = []
        percent = int(self._total_edges()/10)
        self.log("Generating all {} negative training examples".format(self._total_edges(IG)))
        for ni in IG.node:
            for nj in IG.node:
                if ni==nj or IG.has_edge(ni, nj): continue
                if len(examples) % percent == 0:
                    self.log("\t...{}% progress".format((len(examples)/percent)*10))
                examples.append((ni, nj))
        return examples

    def nrandom_negative_examples(self, m, IG):
        """
        :return: list of m, (u,v) randomly selected negative examples (edges that don't exist in the graph)
        """
        examples = []
        self.log("Generating {} negative training examples".format(m))
        percent = int(m/10)
        while len(examples) < m:
            u, v = random.sample(IG.node, 2)
            if IG.has_edge(u,v):  continue
            examples.append((u,v))
            if len(examples) % percent == 0:
                self.log("\t...{}% progress".format((len(examples)/percent)*10))
        self.log("done")
        return examples

    def log(self, *args, **kwargs):
        if self.verbose: logging.info(*args, **kwargs)

    def _ensure_dir_exists(self, path):
        if not os.path.exists(path): os.makedirs(path)

    def load_cache_features_successful(self):
        self.log("Loading features")
        features_path = os.path.join(self.basepath, "data", "features")
        self._ensure_dir_exists(features_path)

        self.log("\tloading train.pickle")
        train_path = os.path.join(features_path, "train.pickle")
        if not os.path.exists(train_path): return False
        self.train_data = cPickle.load(open(train_path))

        self.log("\tloading validation.pickle")
        validation_path = os.path.join(features_path, "validation.pickle")
        if not os.path.exists(validation_path): return False
        self.validation_data = cPickle.load(open(validation_path))

        self.log("\tloading test.pickle")
        test_path = os.path.join(features_path, "test.pickle")
        if not os.path.exists(test_path): return False
        self.test_data = cPickle.load(open(test_path))

        return True

    def load_cached_examples_succesful(self):
        self.log("Loading examples")
        features_path = os.path.join(self.basepath, "data", "features")
        self._ensure_dir_exists(features_path)

        self.log("\tloading pos_examples.pickle")
        pos_path = os.path.join(features_path, "pos_examples.pickle")
        if not os.path.exists(pos_path): return False, None, None
        pos = cPickle.load(open(pos_path))

        self.log("\tloading neg_examples.pickle")
        neg_path= os.path.join(features_path, "neg_examples.pickle")
        if not os.path.exists(neg_path): return False, None, None
        neg = cPickle.load(open(neg_path))

        return True, pos, neg

    def _cache_examples(self, allpos, allneg):
        self.log("Caching nonsplit features in permanent storage")
        features_path = os.path.join(self.basepath, "data", "features")
        pos_path = os.path.join(features_path, "pos_examples.pickle")
        cPickle.dump(allpos, open(pos_path, 'wb'))
        neg_path = os.path.join(features_path, "neg_examples.pickle")
        cPickle.dump(allneg, open(neg_path, 'wb'))

    def _cache_features(self):
        self.log("Caching features in permanent storage")
        features_path = os.path.join(self.basepath, "data", "features")

        self.log("\tDumping train.pickle")
        train_path = os.path.join(features_path, "train.pickle")
        cPickle.dump(self.train_data, open(train_path, 'wb'))

        self.log("\tDumping validation.pickle")
        validation_path = os.path.join(features_path, "validation.pickle")
        cPickle.dump(self.validation_data, open(validation_path, 'wb'))

        self.log("\tDumping test.pickle")
        test_path = os.path.join(features_path, "test.pickle")
        cPickle.dump(self.test_data, open(test_path, 'wb'))

    def _delete_edges(self, ebunch, IG):
        IG.remove_edges_from(ebunch)
        return IG

    def _generate_examples(self, scale, IG, balanced):
         # Positive examples
        npos = len(IG.edges())
        pos = list(IG.edges())
        # Randomize Positive
        random.shuffle(pos)
        # Select Subset
        pos = pos[0:int(npos*scale)]
        m = len(pos)

        # Negative examples
        if balanced:
            neg = self.nrandom_negative_examples(m, IG)
        else:
            nnegative = self._num_neg_needed_to_calibrate_dataset(m, IG)
            neg = self.nrandom_negative_examples(nnegative, IG)

        return pos, neg

    def _positive_rate(self, IG):
        total_pos = len(IG.edges())
        total_neg = self._total_edges(IG)-total_pos
        return total_pos / float(total_neg+total_pos)


    def _split(self, pos, neg, ptrain, pvalidation, IG, balanced):
        idx_train, idx_valid = int(len(pos)*ptrain), int(len(pos)*pvalidation)
        postr, posvalid, postst = pos[0:idx_train], pos[idx_train:idx_train+idx_valid], pos[idx_train+idx_valid: len(pos)]

        if balanced:
            negtr, negvalid, negtst = neg[0:idx_train], neg[idx_train:idx_train+idx_valid], neg[idx_train+idx_valid: len(neg)]

        else:
            # Negative Examples   Generate more to keep original distribution if needed
            totalnegs = self._num_neg_needed_to_calibrate_dataset(len(pos), IG)
            if len(neg) <  totalnegs:
                neg += self.nrandom_negative_examples(totalnegs - len(neg), IG)
            idx_train = self._num_neg_needed_to_calibrate_dataset(len(postr), IG)
            idx_valid = self._num_neg_needed_to_calibrate_dataset(len(posvalid), IG)
            negtr, negvalid, negtst = neg[0:idx_train], neg[idx_train:idx_train+idx_valid], neg[idx_train+idx_valid: len(neg)]

        return (postr, negtr), (posvalid, negvalid), (postst, negtst)

    def preprocess(self, ptrain=.8, pvalidation=.05, use_cache_features=True, use_cache_examples=True, scale=.05,
                   balanced=True):
        """
        Generates features, randomly splits datasets into train, validation, test, and fits classifier.
        Suppose there are m' edges in the dataset, then we generate m=scale*m' positive training examples and m negative training examples. This function sets
        self.train_data to have m*ptrain positive examples and m*ptrain negative examples (to mantain class balance),
        similarly it sets self.test_data to have m*(1-ptrain) positive examples and m*(1-ptrain) negative examples.
        """
        assert pvalidation + ptrain < 1; assert scale <= 1;

        # Try loading feature matrices
        if use_cache_features and self.load_cache_features_successful(): return

        # Copy influence graph
        IGcp = self.IG.copy()

        # Generate Examples
        # Try loading randomized examples
        if use_cache_examples:
            loadsucess, pos, neg = self.load_cached_examples_succesful()
            if not loadsucess:
                self.log("Loading examples failed")
                pos, neg = self._generate_examples(scale, IGcp, balanced=balanced)
                self._cache_examples(pos, neg)
        else:
            pos, neg = self._generate_examples(scale, IGcp, balanced=balanced)

        # Randomize
        self.log("Randomizing")
        np.random.shuffle(pos)
        np.random.shuffle(neg)

        # Split
        self.log("Splitting")
        (postr,negtr), (posvalid, negvalid), (postst, negtst) = self._split(pos, neg, ptrain, pvalidation, IG, balanced)

        # Compute Train Set Class Weights (for cost-sensitive learning)
        posrate = len(postr) / float(len(postr) + len(negtr))
        c1, c0 = 1.0 / posrate, 1.0 / (1-posrate)
        weights = [c0, c1]

        # Delete Test Edges from Graph
        IGcp = self._delete_edges(posvalid+postst+negvalid+negtst, IGcp)     # Note this mutates
        self.featurizer.set_graph(IGcp)                      # featurizer should use pruned graph


        # Generate Features
        postr = zip(postr, self.featurizer.feature_matrix(postr))
        posvalid = zip(posvalid, self.featurizer.feature_matrix(posvalid))
        postst = zip(postst, self.featurizer.feature_matrix(postst))
        negtr = zip(negtr, self.featurizer.feature_matrix(negtr))
        negvalid = zip(negvalid, self.featurizer.feature_matrix(negvalid))
        negtst = zip(negtst, self.featurizer.feature_matrix(negtst))

        # Set data attributes
        # Train
        Xtr = postr + negtr
        Ytr = [1 for _ in xrange(len(postr))] + [0 for _ in xrange(len(negtr))]
        self.log("\tTrain Set has PosRate: {}".format(len(postr) / float(len(Xtr))))
        self.train_data = (Xtr, Ytr)
        # Validation
        Xvalid = posvalid + negvalid
        Yvalid  = [1 for _ in xrange(len(posvalid))] + [0 for _ in xrange(len(negvalid))]
        self.log("\tValidation Set has PosRate: {}".format(len(postr) / float(len(Xtr))))
        self.validation_data = (Xvalid, Yvalid)
        # Test
        Xtst = postst + negtst
        Ytst  = [1 for _ in xrange(len(postst))] + [0 for _ in xrange(len(negtst))]
        self.log("\tTest Set has PosRate: {}".format(len(postr) / float(len(Xtr))))
        self.test_data = (Xtst, Ytst)

        if use_cache_features: self._cache_features()

        # Return class weights
        self.log("Class Weights in Train Set: c0={}, c1={}".format(*weights))
        return weights


    def tune_model(self):
        """
        TODO use validation set to tune hyperparameters
        :return:
        """
        pass

    def fit(self, class_weights):
        # Fit
        self.log("Fitting Model")
        exs, phi = zip(*self.train_data[0])  # train examples and feature mappings
        y = self.train_data[1]

        c0, c1 = class_weights
        self.classifier.fit(phi, y=y, sample_weight=[c0 if yi == 0 else 10*c1 for yi in y])
        self.log("done")

    def predict(self, u, v):
        """
        :return: Returns {1,0} if there should be an influence edge between u,v
        """
        features = self.featurizer.compute_features(u,v)
        return self.classifier.predict(features)

    def predict_from_features(self, feats):
        return self.classifier.predict(feats)

    def predict_proba_from_features(self, feats):
        return self.classifier.predict_proba(feats)[0][1]

    def _num_neg_needed_to_calibrate_dataset(self, npos_in_set, IG):
        """
        Get number negative examples needed to ensure that the overall distribution of test set is the same as the original
        """
        # Compute number of negative examples needed to calibrate the ratio
        total_pos = len(IG.edges())
        total_neg = self._total_edges(IG)-total_pos
        numneg_needed = int(total_neg * npos_in_set / float(total_pos))
        return numneg_needed

    def precision_topk(self, ys, ypreds, class_weights, suffix):
        ypreds = np.asarray(ypreds)
        ys = np.asarray(ys)
        c0, c1 = class_weights
        posrate = 1.0 / c1
        maxk = int(posrate*len(ys))  # Max topk is 2 times the distribution we'd expect
        maxkpow =  math.ceil(np.log10(2*maxk))

        # Sample k values on a log scale.
        # ks = np.array(list(set(map(int, np.logspace(0, maxkpow, num=100)))))

        # Use all k values; don't discard any.
        ks = np.array(range(1, maxk + 1))

        precisions_at_k = []
        for k in ks:
            # Indices of the k examples with the highest y predicted value.
            idices = np.argsort(ypreds)[::-1][0:k]

            # The true y labels of these k examples.
            ysk = ys[idices]

            # Append the precision of these top k predictions.
            yscore = np.ones((len(ysk), 1))
            precisions_at_k.append(precision_score(ysk, yscore))

        plt.figure()
        plt.semilogx(ks, precisions_at_k, 'bo', alpha=.9)
        plt.axvline(x=maxk, ymin=0, linewidth=2, color='r', alpha=.2)
        plt.title("Precision at Topk")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(self.basepath, "plots", "ptopk_{}.png".format(suffix)))

    def report_false_positives(self, ys, ypreds, class_weights):
        print "Reporting top false positives"
        c0, c1 = class_weights
        posrate = 1.0 / c1
        maxk = int(posrate*len(ys)) # Number of positive examples

        labels = GraphLoader(verbose=False).get_artist_ids_to_names()

        # Get the top positive predicted examples, in order from highest y predicted value to lowest.
        top_pred_indices = np.argsort(ypreds)[::-1][0:maxk]

        # Among these, identify the examples that were actually negative examples.
        for i, top_pred_index in enumerate(top_pred_indices):
            if ypreds[top_pred_index] <= 0.5:
                break
            if ys[top_pred_index] == 0:
                # We predicted that artist1 influenced artist2, but this was not the case.
                artist1, artist2 = [labels[artist].strip() for artist in self.test_data[0][top_pred_index][0]]
                print("%dth highest-predicted example (y_pred=%f): %s --> %s" % (i, ypreds[top_pred_index], artist1, artist2))


    def report_false_negatives(self, ys, ypreds, class_weights):
        print "Reporting top false negatives"
        c0, c1 = class_weights
        posrate = 1.0 / c1
        maxk = len(ys) - int(posrate*len(ys)) # Number of negative examples

        labels = GraphLoader(verbose=False).get_artist_ids_to_names()

        # Get the most negatively predicted examples, in order from lowest y predicted value to highest.
        top_pred_indices = np.argsort(ypreds)[0:maxk]

        # Among these, identify the examples that were actually positive examples.
        for i, top_pred_index in enumerate(top_pred_indices):
            if ypreds[top_pred_index] >= 0.5:
                break
            if ys[top_pred_index] == 1:
                # Artist1 influenced artist2, but we predicted otherwise.
                artist1, artist2 = [labels[artist].strip() for artist in self.test_data[0][top_pred_index][0]]
                print("%dth lowest-predicted example (y_pred=%f): %s --> %s" % (i, ypreds[top_pred_index], artist1, artist2))

    def print_feature_weights(self):
        importances = self.classifier.feature_importances_

        # Unused variable. Could use this to create a bar plot showing feature importances with std dev interval.
        # See: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
        std = np.std([tree.feature_importances_ for tree in self.classifier.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        self.log("Ranked feature importances:")

        for f in range(importances.shape[0]):
            self.log("%d. Feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    def auc_metrics(self, y, ypreds):
        print("Features Used: {}".format(self.featurizer.get_feature_names()))
        auc = round(roc_auc_score(y, ypreds), 3)
        print("\tROC AUC: {}".format(auc))
        prc = round(average_precision_score(y, ypreds, average="weighted"), 3)
        print("\tPrecision-Recall AUC: {}".format(prc))
        return auc, prc

    def make_predictions(self):
        exs, phis = zip(*self.test_data[0])
        ys = self.test_data[1]
        self.log("Evaluating Model")
        self.log("Will make {} predictions".format(len(ys)))
        percent = int(len(ys)/10)
        ypreds = []
        for i, phi in enumerate(phis):
            ypreds.append(self.predict_proba_from_features(phi))
            if i % percent == 0:
                self.log("\t...{}% progress".format((i/percent)*10))
        return ys, ypreds

def run(IG, features_to_use, scale=1.0, verbose=True, balanced=True, use_cache_examples=True):
    """
    Train Learner, Make Predictions, Show AUC metrics, plot
    """
    ep = EdgePredictor(IG, verbose=verbose, features_to_use=features_to_use)
    class_weights = ep.preprocess(use_cache_features=False, use_cache_examples=use_cache_examples, balanced=balanced, scale=scale)

    # Fit
    ep.fit(class_weights)

    if verbose:
        ep.print_feature_weights()

    # Make Predictions
    ys, ypreds = ep.make_predictions()

    # AUC Metrics
    aucs, prcs = ep.auc_metrics(ys, ypreds)

    # Topk Metrics
    ep.precision_topk(ys, ypreds, class_weights, '_'.join(ep.featurizer.get_feature_names()))

    if verbose:
        ep.report_false_positives(ys, ypreds, class_weights)
        ep.report_false_negatives(ys, ypreds, class_weights)

    return aucs, prcs

def run_each_feature_independently(IG, verbose=True):
    features = ["rnd", "nc", "jc", "aa", "pa", "ra", "si", "lh", "ja", "da"]
    for f in features:
        run(IG, [f], verbose=verbose)

def run_each_pair_of_features(IG, verbose=True):
    features = ["rnd", "nc", "jc", "aa", "pa", "ra", "si", "lh", "ja", "da"]
    for i1, f1 in enumerate(features):
        for i2, f2 in enumerate(features):
            if i2 > i1:
                run(IG, [f1, f2], verbose=verbose)

def cross_validate(k, features,  scale=1.0, balanced=True, use_cache_examples=True):
    logging.info("Will run {}-fold bootstrap sampling validation".format(k))
    # Load IG graph
    IG = GraphLoader(verbose=True).load_networkx_influence_graph(pruned=False)


    scores = []  # List tuples with elements (roc_auc, precision_recall_auc)
    for _ in range(k):
        logging.info("\tStarting {}th run".format(_))
        auc, prc = run(IG, features, scale=scale, balanced=balanced,
                       use_cache_examples=use_cache_examples)
        scores.append((auc, prc))

    aucs, prcs = zip(*scores)
    print("Averaged ROC AUC {}".format(sum(aucs) / float(len(aucs))))
    print("Averaged PR AUC {}".format(sum(prcs) / float(len(prcs))))


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(format="[%(name)s %(asctime)s]\t%(msg)s", level=logging.INFO)

    # Load IG graph
    IG = GraphLoader(verbose=False).load_networkx_influence_graph(pruned=False)

    # run_each_feature_independently(IG, verbose=False)
    # run_each_pair_of_features(IG, verbose=False)

    cross_validate(5, ["pa"], use_cache_examples=False)
    #run(IG, ["da", "yd", "pa"], scale=1.0, verbose=True)
    # run(IG, ["yd"], scale=1.0, verbose=False)
    # run(IG, ["pr"], scale=1.0, verbose=False)
    # run(IG, ["da", "yd"], scale=1.0, verbose=False)
    # run(IG, ["da", "pr"], scale=1.0, verbose=False)

