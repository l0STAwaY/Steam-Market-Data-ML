class NaiveBayesClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(NaiveBayesClassifier, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes