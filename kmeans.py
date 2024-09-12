import math
import random
from typing import List

class KMeansClusterClassifier:
    centerL = []
    labelL = []
    data_list = []
    
    def __init__(self, n_cluster: int):
        
        self.n_cluster = n_cluster
    
    def fit(self, X: List[List[float]], y: List[int] = None):
        #Randomly initialize cluster centers
        self.centers = random.sample(X, self.n_cluster)
        self.labels = [0] * len(X)
        
        for _ in range(200):  # Maximum number of iterations
            new_labels = [self._closest_center(x) for x in X]
            
            #If labels don't change,break the loop
            if new_labels == self.labels:
                break
            self.labels = new_labels
            
            #Update cluster centers
            for i in range(self.n_cluster):
                points = [X[j] for j in range(len(X)) if self.labels[j] == i]
                if points:
                    self.centers[i] = self._compute_center(points)
        
        self.centerL = self.centers
        self.labelL = self.labels
        self.data_list = X
    
    def predict(self, X: List[List[float]]):
        return [self._closest_center(x) for x in X]
    
    def _closest_center(self, x: List[float]) -> int:
        distances = [self._euclidean_distance(x, center) for center in self.centers]
        return distances.index(min(distances))
    
    def _euclidean_distance(self, x1: List[float], x2: List[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
    
    def _compute_center(self, points: List[List[float]]) -> List[float]:
        return [sum(coord) / len(points) for coord in zip(*points)]

    
    