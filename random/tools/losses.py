from __future__ import absolute_import
import oneflow as flow
import oneflow.nn as nn
import numpy as np
import oneflow.math as math
__all__ = ['TripletLoss']

def addmm(mat,mat1,mat2,beta=1,alpha=1):    
    temp=np.matmul(mat,mat2)
    out=(beta*mat+alpha*temp)
    return out
def _MarginRankingLoss(input1, input2, target, margin = 0, reduction='mean'):
    low_bound = flow.constant_like(target, 0, dtype= flow.float32)

    if reduction == 'none':
        ret = math.maximum(low_bound,
                        math.add(margin,math.multiply(target,math.multiply(-1,math.subtract(input1,input2)))))
    else:
        ret = math.reduce_mean(math.maximum(low_bound,
                        math.add(margin,math.multiply(target,math.multiply(-1,math.subtract(input1,input2))))))
    return ret
class TripletLoss(object):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, distance='euclidean'):
        super(TripletLoss, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.ranking_loss = _MarginRankingLoss
    def build(self,inputs,targets):
        n=inputs.shape[0]
        if self.distance=='euclidean':
            dist=flow.math.pow(inputs,2)
            dist=flow.math.reduce_sum(dist, axis=1, keepdims=True)
            dist=np.tile(dist,(n, n))
            dist_t=flow.transpose(dist)
            dist=dist+dist_t
            inputs_t=flow.transpose(inputs)
            dist=addmm(dist,inputs,inputs_t,beta=1,alpha=-2)
            dist=flow.clamp(min_value=1e-12)
            dist=flow.math.sqrt(dist)
        elif self.distance == 'cosine':
            fnorm=np.linalg.norm(inputs,ord=2,axis=1,keepdims=True)
            l2norm=np.tile(inputs,(inputs.shape))
            l2norm=inputs/l2norm
            l2norm_t=flow.transpose(l2norm)
            dist=-np.matmul(l2norm,l2norm_t)
        target_expand=np.tile(targets,(n,n))
        target_expand_t=flow.transpose(target_expand)
        mask=flow.math.equal(target_expand,target_expand_t)
        dist_ap, dist_an = [], []
        for i in range(n):
            temp=np.ndarray.max(dist[i][mask[i]])
            temp=flow.expand_dims(temp,axis=0)
            dist_ap.append(temp)
            temp=np.ndarray.min(dist[i][mask[i]==0])
            temp=flow.expand_dims(temp,axis=0)
            dist_an.append(temp)
            dist_ap=flow.concat(dist_ap)
            dist_an=flow.concat(dist_an)
        y=flow.ones_like(dist_an)
        loss=self.ranking_loss(dist_an, dist_ap, y,margin=self.margin)
        return loss
        





