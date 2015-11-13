from featurizer import FeatureGenerator
import logging
from loader import GraphLoader
import random
import cPickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics.ranking import roc_auc_score
import os

class EdgePredictor(object):
    # FeatureGenerator from featurizer.py
    featurizer = None

    # Influence graph
    IG = None

    # For now just easy ensemble method
    classifier = ExtraTreesClassifier()

    # Tuple Xtrain, Phitrain, Ytrain, set when self.train() is called by the client
    train_data = None

    # Tuple Xvalid, Phivalid, Yvalid set when self.train() is called by the client
    validation_data = None

    # Tuple Xtst, Phitst, Ytst set when self.train() is called by the client
    test_data = None


    def __init__(self, IG, verbose=True):
        random.seed(0)
        self.basepath = os.path.dirname(os.path.dirname(__file__))
        self.IG = IG
        self.featurizer = FeatureGenerator(IG)
        self.verbose = verbose

    def all_negative_examples(self):
        examples = []
        total_edges = len(self.IG.nodes()) * (len(self.IG.nodes())-1) / 2.0
        percent = int(total_edges/10)
        self.log("Generating all {} negative training examples".format(total_edges))
        for ni in self.IG.node:
            for nj in self.IG.node:
                if ni==nj or self.IG.has_edge(ni, nj): continue
                if len(examples) % percent == 0:
                    self.log("\t...{}% progress".format((len(examples)/percent)*10))
                examples.append((ni, nj))
        return examples

    def nrandom_negative_examples(self, m):
        """
        :return: list of m, (u,v) randomly selected negative examples (edges that don't exist in the graph)
        """
        examples = []
        self.log("Generating {} negative training examples".format(m))
        percent = int(m/10)
        while len(examples) < m:
            u, v = random.sample(self.IG.node, 2)
            if self.IG.has_edge(u,v):  continue
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

    def load_cache_successful(self):
        features_path = os.path.join(self.basepath, "data", "features")
        self._ensure_dir_exists(features_path)

        train_path = os.path.join(features_path, "train.pickle")
        if not os.path.exists(train_path): return False
        self.train_data = cPickle.load(open(train_path))

        validation_path = os.path.join(features_path, "validation.pickle")
        if not os.path.exists(validation_path): return False
        self.validation_data = cPickle.load(open(validation_path))

        test_path = os.path.join(features_path, "test.pickle")
        if not os.path.exists(test_path): return False
        self.test_data = cPickle.load(open(test_path))

        return True

    def _cache_features(self):
        features_path = os.path.join(self.basepath, "data", "features")

        train_path = os.path.join(features_path, "train.pickle")
        cPickle.dump(self.train_data, open(train_path, 'wb'))

        validation_path = os.path.join(features_path, "validation.pickle")
        cPickle.dump(self.validation_data, open(validation_path, 'wb'))

        test_path = os.path.join(features_path, "test.pickle")
        cPickle.dump(self.test_data, open(test_path, 'wb'))

    def train(self, ptrain=.6, pvalidation=.2, use_cache=True):
        """
        Generates features, randomly splits datasets into train, validation, test, and fits classifier.
        Suppose there are m' edges in the dataset, then we generate m=size*m' positive training examples and m negative training examples. This function sets
        self.train_data to have m*ptrain positive examples and m*ptrain negative examples (to mantain class balance),
        similarly it sets self.test_data to have m*(1-ptrain) positive examples and m*(1-ptrain) negative examples.
        :param ptrain: The percentage of the pruned dataset to use for training
        :param pvalidation: The percentage to use as validation set
        """
        assert pvalidation + ptrain < 1;
        if use_cache and self.load_cache_successful(): return

        # Generate positive, negative examples
        pos = self.IG.edges()
        phipos = self.featurizer.feature_matrix(self.IG.edges())
        pos = zip(pos, phipos)
        m = len(pos)

        allneg = self.all_negative_examples()
        neg = allneg[0:m]  #  Just grab the first m for now
        phineg = self.featurizer.feature_matrix(neg)
        neg = zip(neg, phineg)

        # Randomize
        random.shuffle(pos)
        random.shuffle(neg)

        # Split
        postr, posvalid, postst = self._split(pos, ptrain, pvalidation)  # positive train, positive test
        negtr, negvalid, negtst = self._split(neg, ptrain, pvalidation)  # negative train, negative test

        # Set Attributes
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
        Ytst  = [1 for _ in xrange(len(postst))] + [0 for _ in xrange(len(negvalid))]
        Xtst += allneg[m:len(allneg)]
        Ytst = [0 for _ in xrange(len(Xtst) - len(Ytst))]
        self.tst_data = (Xtst, Ytst)

        if use_cache: self._cache_features()

    def fit(self):
        # Fit
        self.log("Fitting Model")
        exs, phi = zip(*self.train_data[0])  # train examples and feature mappings
        self.classifier.fit(phi, self.train_data[1])
        self.log("done")

    def predict(self, u, v):
        """
        :return: Returns {1,0} if there should be an influence edge between u,v
        """
        features = self.featurizer.get_features(u,v)
        return self.classifier.predict(features)

    def predict_from_features(self, feats):
        return self.classifier.predict(feats)

    def _calibrate_test_data(self):
        """
        Sets negative examples to train data to ensure that the overall distribution is the same as the original
        """
        # Compute number of negative examples needed to calibrate the ratio
        pos_in_tst = len(self.test_data)
        total_edges = len(self.IG.nodes()) * (len(self.IG.nodes())-1) / 2.0
        total_pos = len(self.IG.edges())
        total_neg = total_edges-total_pos
        numneg_needed = total_neg * pos_in_tst / float(total_pos)

        # Get unused negative examples
        Xtst, Phitst, Ytst = self._get_unused_negative_examples(numneg_needed)

        # Set self.test_data to account for calibration
        self.test_data[0] += zip(Xtst, Phitst)
        self.test_data[1] += Ytst

    def evaluate_model(self):
        exs, phis = zip(*self.test_data[0])
        ys = self.test_data[1]
        self.log("Evaluating Model")
        self.log("Will make {} predictions".format(len(ys)))
        percent = int(len(ys)/10)
        ypreds = []
        for i, phi in enumerate(phis):
            ypreds.append(self.predict_from_features(phi))
            if i % percent == 0:
                self.log("\t...{}% progress".format((i/percent)*10))
        print roc_auc_score(ys, ypreds)

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(format="[%(name)s %(asctime)s]\t%(msg)s", level=logging.INFO)

    # Load IG graph
    IG = GraphLoader(verbose=True).load_networkx_influence_graph(pruned=False)

    # Initialize and train Predictor
    ep = EdgePredictor(IG)
    ep.train()

    # Evaluate
    ep.evaluate_model()

