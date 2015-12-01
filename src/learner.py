from featurizer import FeatureGenerator
import logging
from loader import GraphLoader
import random
import cPickle
import numpy as np
import os
import warnings
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm

with warnings.catch_warnings():
    # sklearn is complaining about having 1 feature
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.metrics.ranking import roc_auc_score, average_precision_score

class EdgePredictor(object):
    # FeatureGenerator from featurizer.py
    featurizer = None

    # Influence graph
    IG = None

    # For now just easy ensemble method
    classifier = svm.SVC(class_weight="balanced")
    #classifier = ExtraTreesClassifier()

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
        self.featurizer = FeatureGenerator(features_to_use=features_to_use)
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

    def _split(self, x, ptrain, pvalidation):
        idx_train, idx_valid = int(len(x)*ptrain), int(len(x)*pvalidation)
        return x[0:idx_train], x[idx_train:idx_train+idx_valid], x[idx_train+idx_valid: len(x)]

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

    def _generate_examples(self, scale, IG, ptrain, pvalidation):
         # Positive examples
        npos = len(IG.edges())
        pos = list(IG.edges())
        # Randomize Positive
        random.shuffle(pos)
        # Select Subset
        pos = pos[0:int(npos*scale)]
        m = len(pos)

        # Negative examples
        if scale == 1:
            allneg = self.all_negative_examples(IG)  # Faster than self.nrandom_negative_examples when scale ~ 1
        else:
            nnegative = self._num_neg_needed_to_calibrate_dataset(m, IG)
            allneg = self.nrandom_negative_examples(nnegative, IG)

        return pos, allneg

    def _positive_rate(self, IG):
        total_pos = len(IG.edges())
        total_neg = self._total_edges(IG)-total_pos
        return total_pos / float(total_neg+total_pos)

    def preprocess(self, ptrain=.8, pvalidation=.05, use_cache_features=True, use_cache_examples=True, scale=.0005):
        """
        Generates features, randomly splits datasets into train, validation, test, and fits classifier.
        Suppose there are m' edges in the dataset, then we generate m=scale*m' positive training examples and m negative training examples. This function sets
        self.train_data to have m*ptrain positive examples and m*ptrain negative examples (to mantain class balance),
        similarly it sets self.test_data to have m*(1-ptrain) positive examples and m*(1-ptrain) negative examples.

        Note we are implicitly over-sampling the minority class (this is one way to combat class imbalance) but we need
        to make sure that we are testing on the original distribution. TODO other ways to tackle class imbalance:
        using cost-sensitive learning (adding class weights to classifiers), generate synthetic samples (using SMOTE),
        frame is as outlier detection (use once-class SVM)
        :param ptrain: The percentage of the pruned dataset to use for training
        :param pvalidation: The percentage to use as validation set
        :scale: The scale of the dataset to use, needed because running the entire dataset is too slow (4 hours)
        """
        assert pvalidation + ptrain < 1; assert scale < 1;

        # Try loading feature matrices
        if use_cache_features and self.load_cache_features_successful(): return

        # Copy influence graph
        IGcp = self.IG.copy()
        posrate = self._positive_rate(IGcp)

        # Compute Class Weights (for cost-sensitive learning)
        c1, c0 = 1.0 / posrate, 1.0 / (1-posrate)
        weights = [c0, c1]

        # Generate Examples
        # Try loading randomized examples
        if use_cache_examples:
            loadsucess, pos, allneg = self.load_cached_examples_succesful()
            if not loadsucess:
                self.log("Loading examples failed")
                pos, allneg = self._generate_examples(scale, IGcp, ptrain, pvalidation)
                self._cache_examples(pos, allneg)
        else:
            pos, allneg = self._generate_examples(scale, IGcp, ptrain, pvalidation)

        # Select subset
        neg = allneg[0:len(pos)]

        # Split
        self.log("Splitting")
        postr, posvalid, postst = self._split(pos, ptrain, pvalidation)  # positive train, positive test
        negtr, negvalid, negtst = self._split(neg, ptrain, pvalidation)  # negative train, negative test

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
        self.train_data = (Xtr, Ytr)
        # Validation
        Xvalid = posvalid + negvalid
        Yvalid  = [1 for _ in xrange(len(posvalid))] + [0 for _ in xrange(len(negvalid))]
        self.validation_data = (Xvalid, Yvalid)
        # Test
        Xtst = postst + negtst
        Ytst  = [1 for _ in xrange(len(postst))] + [0 for _ in xrange(len(negtst))]
        numneg = self._num_neg_needed_to_calibrate_dataset(len(postst), IGcp)
        Xtst += zip(allneg[len(pos):numneg-len(negtst)], self.featurizer.feature_matrix(allneg[len(pos):numneg-len(negtst)]))
        Ytst += [0 for _ in xrange(len(Xtst) - len(Ytst))]
        print("Ratio pos/all in tst: {}".format(float(len(postst)) / float(len(Xtst))))
        self.test_data = (Xtst, Ytst)

        if use_cache_features: self._cache_features()

        # Return class weights
        return weights

    def tune_model(self):
        """
        TODO use validation set to tune hyperparameters
        :return:
        """
        pass

    def fit(self):
        # Fit
        self.log("Fitting Model")
        exs, phi = zip(*self.train_data[0])  # train examples and feature mappings
        phis_ys = zip(phi, self.train_data[1])
        random.shuffle(phis_ys)
        phi, y = zip(*phis_ys)
        self.classifier.fit(np.asarray(phi).reshape(-1,1), np.asarray(y))
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
        self.log("Original Ration Pos/All: {}".format(total_pos / float(total_neg+total_pos)))
        numneg_needed = int(total_neg * npos_in_set / float(total_pos))
        return numneg_needed

    def evaluate_model(self):
        exs, phis = zip(*self.test_data[0])
        ys = self.test_data[1]
        self.log("Evaluating Model")
        self.log("Will make {} predictions".format(len(ys)))
        percent = int(len(ys)/10)
        ypreds = []
        for i, phi in enumerate(phis):
            ypreds.append(self.predict_proba_from_features(phi))
            #ypreds.append(0)
            if i % percent == 0:
                self.log("\t...{}% progress".format((i/percent)*10))
        print("Features Used: {}".format(self.featurizer.get_feature_names()))
        print("\tROC AUC: {}".format(roc_auc_score(ys, ypreds)))
        # Note that for this problem the PRAUC as below is probably more informative but let's just save that
        # for the final project report
        print("\tPrecision-Recall AUC: {}".format(average_precision_score(ys, ypreds, average="weighted")))

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(format="[%(name)s %(asctime)s]\t%(msg)s", level=logging.INFO)

    # Load IG graph
    IG = GraphLoader(verbose=True).load_networkx_influence_graph(pruned=False)

    # Initialize and train Predictor
    features = ["nc", "jc", "aa", "pa", "ra", "si", "lh", "rdn"] # This list here for reference
    ep = EdgePredictor(IG, features_to_use=["nc"])
    ep.preprocess(use_cache_features=False, use_cache_examples=False)

    # Fit
    ep.fit()

    # Evaluate
    ep.evaluate_model()

