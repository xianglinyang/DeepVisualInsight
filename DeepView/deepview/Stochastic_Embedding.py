# Based on code of Leland McInnes
#
# License: BSD 3 clause



from numba import jit, prange
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay

from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve as lin_solve

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf

def euclid(x_,r_,s_,sigma_,nn_,a=1,b=1, y_=None, H_=None):
    n_execs, n_dim = nn_.shape[0], s_.shape[1]
    
    if y_ is None:
        y_ = np.zeros( ( n_execs,n_dim ) , dtype=np.float32 )
    else:
        for k in range(n_execs):
            for j in range(n_dim):
                y_[k,j] = 0
    if H_ is None:
        H_ = np.zeros( (n_execs) , dtype=np.float32 )
    else:
        for k in range(n_execs):
            H_[k] = 0
    
    return euclid__(x_,r_,s_,sigma_,nn_,a,b, y_, H_)

@jit(nopython=True, fastmath=True, parallel=True)
def euclid__(x_,r_,s_,sigma_,nn_,a,b, y_, H_):
    n_execs, n_samps, m_dim, n_dim = nn_.shape[0], nn_.shape[1], r_.shape[1], s_.shape[1]
    
    for k in range(n_execs):
        w_ = np.zeros( (n_samps) , dtype=np.float32 )
        w_i_div_sigma_i_sum = 0
        for samp in range(n_samps):
            i = nn_[k,samp]
            dist = 0
            for j in range(m_dim):
                dist += (x_[k,j]-r_[i,j])**2
            w_[samp] = 1/(1+a*dist**b)
            w_i_div_sigma_i = w_[samp] / sigma_[i]
            w_i_div_sigma_i_sum += w_i_div_sigma_i
            for j in range(n_dim):
                y_[k,j] += w_i_div_sigma_i*s_[i,j]
        for j in range(n_dim):
            y_[k,j] /= w_i_div_sigma_i_sum
        for samp in range(n_samps):
            dist = 0
            for j in range(n_dim):
                dist += (y_[k,j]-s_[i,j])**2
            logv_i = -dist/sigma_[samp]
            w_i = w_[samp]
            H_[k] += w_i*logv_i + (1-w_i)*(1-np.exp(logv_i))
    return y_, H_

@jit(nopython=True, fastmath=True, parallel=True)
def weight__(x_,r_,sigma_,nn_,a,b):
    n_execs, n_samps, m_dim = nn_.shape[0], nn_.shape[1], r_.shape[1]
    
    w_ = np.zeros( (n_execs,n_samps) , dtype=np.float32 )
    for k in range(n_execs):
        w_i_div_sigma_i_sum = 0
        for samp in range(n_samps):
            i = nn_[k,samp]
            dist = 0
            for j in range(m_dim):
                dist += (x_[k,j]-r_[i,j])**2
            w_ki = 1/(1+a*dist**b) / sigma_[i]
            w_i_div_sigma_i_sum += w_ki
            w_[k,samp] = w_ki
        for samp in range(n_samps):
            w_[k,samp] /= w_i_div_sigma_i_sum
    return w_

def compute_H(x_,y_,r_,s_,sigma_,nn_=None,a=1,b=1, H_=None,att_=None,rep_=None, compute_H_only=True):
    n_execs = x_.shape[0]
    if nn_ is None:
        n_samps = r_.shape[0]
        use_all_data = True
        nn_ = np.empty( (0,0), dtype=np.int )
    else:
        n_samps = nn_.shape[1]
        use_all_data = False
    if compute_H_only:
        if H_ is None:
            H_ = np.empty( (n_execs) , dtype=np.float32 )
        if att_ is None:
            att_ = np.empty( (0) , dtype=np.float32 )
        if rep_ is None:
            rep_ = np.empty( (0) , dtype=np.float32 )
    else:
        if H_ is None:
            H_ = np.empty( (0) , dtype=np.float32 )
        if att_ is None:
            att_ = np.empty( (n_execs) , dtype=np.float32 )
        if rep_ is None:
            rep_ = np.empty( (n_execs) , dtype=np.float32 )
    
    res = compute_H__(x_,y_,r_,s_,sigma_,n_samps,use_all_data,nn_,a,b, H_,att_,rep_, compute_H_only)
    
    if compute_H_only:
        return res[0]
    else:
        return res[1],res[2]

@jit(nopython=True, fastmath=True, parallel=True)
def compute_H__(x_,y_,r_,s_,sigma_,n_samps,use_all_data,nn_,a,b, H_,att_,rep_, compute_H_only):
    n_execs, m_dim, n_dim = x_.shape[0], r_.shape[1], s_.shape[1]

    for k in range(n_execs):
        att_k, rep_k = 0,0
        for samp in range(n_samps):
            if use_all_data:
                i = samp
            else:
                i = nn_[k,samp]

            dist = 0
            for j in range(m_dim):
                dist += (x_[k,j]-r_[i,j])**2
            w_i = 1/(1+a*dist**b)

            dist = 0
            for j in range(n_dim):
                dist += (y_[k,j]-s_[i,j])**2
            logv_i = -dist/sigma_[samp]

            att_k, rep_k = att_k+w_i*logv_i , rep_k+(1-w_i)*(1-np.exp(logv_i))
        if compute_H_only:
            H_[k] = att_k+rep_k
        else:
            att_[k], rep_[k] = att_k, rep_k
    
    return  H_, att_, rep_

@jit(nopython=True, fastmath=True, parallel=True)
def smooth(y_,H_,nn_,itrs,in_place=True):
    n_execs, n_dim, n_samps = y_.shape[0], y_.shape[1], nn_.shape[1]

    H_min, H_max = H_.min(), H_.max()
    H_norm = (H_.copy()-H_min)/(H_max-H_min)
    
    if in_place:
        for itr in range(itrs):
            perm = np.random.permutation(n_execs)
            for kp in range(n_execs):
                k = perm[kp]
                H_norm_k = H_norm[k]
                one_H_norm_k = 1-H_norm_k
                for j in range(n_dim):
                    ava_ji = 0
                    for i in range(n_samps):
                        ava_ji += y_[nn_[k,i],j]
                    y_[k,j] = one_H_norm_k*y_[k,j] + H_norm_k*ava_ji/n_samps
        return y_
    else:
        y_a,y_b = y_, np.zeros( y_.shape , dtype=np.float32 )
        for itr in range(itrs):
            for k in prange(n_execs):
                H_norm_k = H_norm[k]
                one_H_norm_k = 1-H_norm_k
                for j in range(n_dim):
                    ava_ji = 0
                    for i in range(n_samps):
                        ava_ji += y_a[nn_[k,i],j]
                    y_b[k,j] = one_H_norm_k*y_a[k,j] + H_norm_k*ava_ji/n_samps
            y_c = y_b
            y_b = y_a
            y_a = y_c
        return y_a

#@jit(parallel=True)
def select_neighbors(candidates):
        n_sets, n_select, n_neighbors = candidates.shape[0], candidates.shape[1], candidates.shape[2]
        
        result = np.zeros( (n_sets, n_neighbors) , dtype=np.int32 )
        k = np.zeros( n_select , dtype=np.int32 )
        for i in range(n_sets):
            cand = set()  
            j, l = 0, 0
            for j_ in range(n_select):
                k[j_] = 0
            while l < n_neighbors:
                nn_id = candidates[i,j,k[j]]
                k[j] += 1
                if not(nn_id in cand):
                    cand.add(nn_id)
                    result[i,l] = nn_id
                    l = l+1
                    j = (j+1)%n_select
        return result

@jit(nopython=True, fastmath=True, parallel=True)
def dist_each(X,Ys,nn):
    n_execs, n_samps, n_dim = nn.shape[0], nn.shape[1], Ys.shape[1]
    dists = np.zeros( (n_execs,n_samps) , dtype=np.float32 )
    for i in range(n_execs):
        x_i = X[i]
        for samp in range(n_samps):
            #k = nn[i,samp]
            y_k = Ys[nn[i,samp]]
            dist = 0
            for d in range(n_dim):
                dist += (x_i[d]-y_k[d])**2
            dists[i,samp] = dist
    return dists

@jit(nopython=True, parallel=True, fastmath=True)
def compute_sigma(distances, k, n_iter=64, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In esscence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.UMAP

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    result = np.zeros(distances.shape[0], dtype=np.float32)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0


            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid
        
        if result[i] < MIN_K_DIST_SCALE * np.mean(distances):
             result[i] = MIN_K_DIST_SCALE * np.mean(distances)

    return result

def find_grid_nn(X,n_neighbors):
    return NearestNeighbors().fit(X).kneighbors(X, n_neighbors=n_neighbors, return_distance=False)
                
class StochasticEmbedding(BaseEstimator):
    
    def __init__(
        self,
        n_neighbors=20,
        n_smoothing_neighbors=None,
        n_smoothing_epochs=25,
        n_centroids=None,
        border_min_dist=1,
        a=None,
        b=None,
        verbose=False,
    ):

        self.n_neighbors = n_neighbors
        self.n_centroids = n_centroids
        self.n_smoothing_neighbors = n_smoothing_neighbors
        self.n_smoothing_epochs = n_smoothing_epochs
        self.border_min_dist = border_min_dist
        self.verbose = verbose
        self.a = a
        self.b = b
        
    def _validate_parameters(self):
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 2")
        if (self.n_centroids is not None) and ((not isinstance(self.n_centroids, int)) or self.n_centroids <= 0):
            raise ValueError("n_centroids must be a positive integer or None for self estimid")
        if self.border_min_dist <= 0:
            raise ValueError("border_min_dist must be greater then zero")
        if (self.n_smoothing_neighbors is not None) and ((not isinstance(self.n_smoothing_neighbors, int)) or self.n_smoothing_neighbors < 2):
            raise ValueError("n_smoothing_neighbors must be a positive integer greater then 2 or None for self estimid")
        
    def _fit(self, X, Y, lab=None, direct_adaption=True, eta=0.1, max_itr=500, F=None):
        """Fit X into an embedded space.

        Optionally use y for supervised dimension reduction.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.

        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.
        """
        X = check_array(X, dtype=np.float32, accept_sparse="csr")
        Y = check_array(Y, dtype=np.float32, accept_sparse="csr")
        
        if len(Y.shape) != 2:
            raise ValueError("Shape missmatch, expected (samples,feature)")
        if len(X.shape) != 2:
            raise ValueError("Shape missmatch, expected (samples,components)")
        
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Shape missmatch, sample")
        
        #self.s_ = Y
        self.r_ = X

        # Handle all the optional arguments, setting default
        self._validate_parameters()
        
        if self.a is None or self.b is None:
            self._a, self._b = np.float32(1), np.float32(1)
        else:
            self._a, self._b = np.float32(self.a), np.float32(self.b)
        
        if self.n_centroids is None:
            if lab is None:
                n_centroids = min(int(np.ceil( Y.shape[0]**0.5 )),Y.shape[0])
            else:
                n = np.unique(lab).shape[0]
                n_centroids = min(int(np.ceil( (Y.shape[0]*n)**0.5 )),Y.shape[0])
        else:
            n_centroids = self.n_centroids
        if self.n_smoothing_neighbors is None:
            self._n_smoothing_neighbors = 2**X.shape[1]+1
        else:
            self._n_smoothing_neighbors = self.n_smoothing_neighbors

        if self.verbose:
            print(str(self))
            
        if X.shape[0] <= self.n_neighbors:
            #warn(
            #    "n_neighbors is larger than the dataset size; truncating to "
            #    "X.shape[0] - 1"
            #)
            self._n_neighbors = X.shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors
        
        
        if self.verbose:
            print("compute triangolation")
        self.min_,self.max_ = np.min(X,axis=0)-self.border_min_dist,np.max(X,axis=0)+self.border_min_dist
        if lab is None:
            centroieds = KMeans(n_clusters=n_centroids).fit(X).cluster_centers_
        else:
            n = lab.max()+1
            labY = np.zeros( (X.shape[0],n) )
            for i in range(lab.shape[0]):
                labY[i,lab[i]] = 10
            centroieds = KMeans(n_clusters=n_centroids).fit(X,y=lab).cluster_centers_
        border = np.array([[self.min_[0],self.min_[1]],[self.max_[0],self.min_[1]],[self.min_[0],self.max_[1]],[self.max_[0],self.max_[1]]])
        self._centroieds = np.concatenate((border,centroieds), axis=0)
        
        if self.verbose:
            print("precompute neighborhood")
        neighbors_search = NearestNeighbors().fit(X)
        self._centroied_neighbors = neighbors_search.kneighbors(self._centroieds, n_neighbors=self._n_neighbors, return_distance=False)
        self._triangolation = Delaunay(self._centroieds)
        
        if self.verbose:
            print("compute sigma")
        """Ynn = neighbors_search.kneighbors(X, n_neighbors=self._n_neighbors, return_distance=False)[:,1:]
        Ydist = dist_each(Y,Y,Ynn)
        self.sigma_ = compute_sigma(Ydist, self._n_neighbors)"""
        self.sigma_ = np.ones(X.shape[0])
        
        if not(direct_adaption):
            self.s_ = Y
        else:
            if self.verbose:
                print("compute s")
            nn = self.find_nn(self.r_)
            w = weight__(self.r_,self.r_,self.sigma_,nn,self._a,self._b)
            
            #lambd = 0.1
            Wp = ([],([],[]))
            #diag = set([i for i in range(nn.shape[0])])
            for i in range(nn.shape[0]):
                for j in range(nn.shape[1]):
                    Wp[1][0].append(i)
                    Wp[1][1].append(nn[i,j])
                    Wp[0].append(w[i,j])
                    #if i == nn[i,j]:
                    #    diag.remove(i)
            #for i in diag:
            #    Wp[1][0].append(i)
            #    Wp[1][1].append(i)
            #    Wp[0].append(0)
            W = csr_matrix( Wp, ( self.r_.shape[0], self.r_.shape[0] ), dtype=np.float32)
            W = W.toarray()
            
            # out = W(x) @ S = (S^T @ W(x)^T)^T
            # Y = W(R) @ S => Y^T = S^T @ W(R)^T
            #              => S = (S^T)^T := (W^T @ W)^-1 @ (W^T @ Y) 
            #              => (W^T @ W) S = W^T @ Y
            
            #W = W + eye(W.shape[0])
            #Wt = W.transpose()
            #G = Wt @ W + lambd * eye(W.shape[0])
            
            #print( W.toarray(), W.toarray().sum(axis=1), (W.transpose() @ W).toarray() )
            #self.s_ = np.array( lin_solve(W.transpose() @ W , W.transpose() @ Y), dtype=np.float32 )
            
            """GT,YT = W.transpose(),Y.transpose()
            print(GT.shape)
            GTI = np.linalg.pinv(GT)
            ST = YT @ GTI
            self.s_ = ST.transpose()"""
            
            """S = np.copy(Y)
            eta = 0.01
            for i in range(1000):
                diff = Y - W @ S
                if i % 10 == 0:
                    err = np.linalg.norm(diff, axis=1)
                    print(err.min(),err.mean(),err.max()," ",err.var(),"   ",diff.shape,W.shape)
                S = S + eta * (W @ diff)
            self.s_ = S"""
            
            
            S = np.copy(Y)
            if F is None:
                euc_itr = max_itr
            else:
                euc_itr = int(max_itr/2)
            
            for i in range(euc_itr):
                E = (W @ S - Y)
                #if i % 50 == 0:
                #    print("eucl err ", np.linalg.norm(E, axis=1).mean())
                S = S - eta * W.transpose() @ E
            
            if not(F is None):
                buf = np.empty( (X.shape[0], Y.shape[1]) )
                for i in range(max_itr):
                    E = W @ S - Y
                    err = 0
                    for k in range(X.shape[0]):
                        nsqerr = np.dot(F[k],E[k])
                        err += np.inner(nsqerr,nsqerr)**0.5
                        buf[k] = np.dot(F[k].transpose(), nsqerr)
                    err /= X.shape[0]
                    #if i % 50 == 0:
                    #    print("eucl err ", np.linalg.norm((W @ S - Y), axis=1).mean(), ", fish err ", err) 
                    grad = W.transpose() @ buf
                    
                    eta_a, eta_b = 0, 0
                    for k in range(X.shape[0]):
                        """print(grad.shape)
                        print(W.shape)
                        print(W[k].shape)
                        print(F[k].shape)
                        print(np.dot(grad.transpose(),W[k].reshape([-1])).shape)
                        print("Ek", E[k].shape) """
                        d = np.dot(F[k],np.dot(grad.transpose(),W[k].reshape([-1])))
                        e = np.dot(F[k],            E[k] )
                        eta_a += np.inner(e,d)
                        eta_b += np.inner(d,d)
                        
                    S = S - eta_a/eta_b * grad
            
            self.s_ = S
            """err = np.linalg.norm(euclid(X,self.r_,Y,self.sigma_,nn,self._a,self._b )[0] - Y, axis=1)
            print(err.min(),err.mean(),err.max()," ",err.var())
            err = np.linalg.norm(euclid(X,self.r_,self.s_,self.sigma_,nn,self._a,self._b )[0] - Y, axis=1)
            print(err.min(),err.mean(),err.max()," ",err.var())"""
        
        return self
    
    def find_nn(self, X):
        return select_neighbors(self._centroied_neighbors[self._triangolation.simplices[self._triangolation.find_simplex(X)]])
    
    def _transform(self,x,return_H=None):
        if self.verbose:
            print("compute initial embedding")
        nn = self.find_nn(x)
        y,H = euclid(x,self.r_,self.s_,self.sigma_,nn,self._a,self._b )
        if self.verbose:
            print("iterative smoothing")
        y = smooth(y,H,find_grid_nn(x, self._n_smoothing_neighbors),self.n_smoothing_epochs,in_place=False)
        if return_H is None:
            return y
        else:
            if self.verbose:
                print("compute H")
            if return_H is "fast":
                return y, compute_H(x,y,self.r_,self.s_,self.sigma_,nn_=nn,a=self._a,b=self._b,H_=H)
            elif return_H is "all":
                return y, compute_H(x,y,self.r_,self.s_,self.sigma_,nn_=None,a=self._a,b=self._b,H_=H)
            else:
                raise ValueError("Unexpected return_H value, possible values are None, 'fast', 'all'")

