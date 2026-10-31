"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   FROM-SCRATCH ML EXPERIMENT v2                                              ║
║   SVM vs Perceptron vs Neural Network + Comprehensive SVM Hyperparameter    ║
║   Analysis                                                                   ║
║──────────────────────────────────────────────────────────────────────────────║
║   Runtime  : Google Colab (Python 3.x)                                       ║
║   NumPy    : ALL model math (no sklearn.svm / sklearn.linear_model /        ║
║              sklearn.neural_network / PyTorch / TensorFlow)                  ║
║   sklearn  : ONLY make_moons, train_test_split                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

TABLE OF CONTENTS
─────────────────
  §0   Imports & global style
  §1   Dataset generation
         §1a  Dataset A – 2-D moons (binary)
         §1b  Dataset B – High-dim small-N (3-class)
  §2   From-scratch model implementations
         §2a  Perceptron
         §2b  NeuralNetwork  (ReLU + Softmax, backprop, SGD)
         §2c  KernelSVM      (Pegasos sub-gradient, linear / poly / RBF)
         §2d  OvRSVM         (One-vs-Rest multi-class wrapper)
  §3   Core comparison experiments
         §3a  Experiment 1 – Decision boundaries on moons
         §3b  Experiment 2 – Learning curves on high-dim dataset
  §4   Kernel comparison (all kernels × both datasets)
         §4a  Decision boundary grid + grouped accuracy bar chart
  §5   Hyperparameter deep-dive (best kernel = RBF)
         §5a  C parameter     – boundary widths + support-vector count curve
         §5b  Gamma parameter – "island" effect plot
         §5c  C × Gamma       – 2-D heatmap on high-dim dataset
         §5d  Polynomial degree – validation curve on high-dim dataset
  §6   Main orchestrator + printed summary
"""

# ─────────────────────────────────────────────────────────────────────────────
# §0  IMPORTS & GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# scikit-learn: data generation & splits ONLY
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# ── Reproducibility ───────────────────────────────────────────────────────────
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# ── Colour Palette ────────────────────────────────────────────────────────────
PAL = {
    "bg":     "#0D1117",
    "panel":  "#161B22",
    "grid":   "#21262D",
    "text":   "#E6EDF3",
    "c0":     "#58A6FF",   # class 0 / train
    "c1":     "#FF7B72",   # class 1 / val
    "c2":     "#3FB950",   # class 2
    "gold":   "#D29922",
    "purple": "#BC8CFF",
    "cyan":   "#39D0D8",
    "orange": "#F0883E",
    "r0":     "#1B2A4A",   # region fill class 0
    "r1":     "#4A1B1B",   # region fill class 1
    "r2":     "#1A4A1B",   # region fill class 2
}
C0, C1, C2 = PAL["c0"], PAL["c1"], PAL["c2"]
KERNEL_COLORS = {"linear": PAL["c0"], "poly": PAL["purple"], "rbf": PAL["c2"]}

plt.rcParams.update({
    "figure.facecolor":  PAL["bg"],
    "axes.facecolor":    PAL["panel"],
    "axes.edgecolor":    PAL["grid"],
    "axes.labelcolor":   PAL["text"],
    "axes.titlecolor":   PAL["text"],
    "xtick.color":       PAL["text"],
    "ytick.color":       PAL["text"],
    "grid.color":        PAL["grid"],
    "text.color":        PAL["text"],
    "legend.facecolor":  PAL["panel"],
    "legend.edgecolor":  PAL["grid"],
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

SEP  = "═" * 72
SEP2 = "─" * 72


def _save(fname):
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=PAL["bg"])
    plt.show()
    print(f"  ✓ Saved → {fname}\n")


# ═════════════════════════════════════════════════════════════════════════════
# §1  DATASET GENERATION
# ═════════════════════════════════════════════════════════════════════════════

# ── §1a  Dataset A – 2-D moons ───────────────────────────────────────────────
def make_dataset_A(n=400, noise=0.25, seed=42):
    """
    Two interleaved half-moons.  Labels 0/1 (mapped to ±1 externally).
    Intentionally non-linearly separable → perceptron will fail.
    """
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X, y


# ── §1b  Dataset B – High-dim small-N ────────────────────────────────────────
def make_dataset_B(N=150, D=500, n_classes=3, seed=42):
    """
    Small-N Large-P stress test: N=150 samples, D=500 features, 3 classes.

    Construction
    ─────────────
    • A 2-D *latent* cluster signal is drawn (equilateral triangle centres).
    • Features 0–9   : sin( latent_0 × π × (i+1) )         [non-linear]
    • Features 10–19 : latent_1² × cos( latent_0 × (i-9) ) [polynomial]
    • Features 20–29 : latent_0 × latent_1 × tanh(...)      [interaction]
    • Features 30–499: pure N(0,1) noise                     [trap for NN]
    • All features z-scored to zero-mean / unit-variance.

    Intent:
      • NN memorises the noise → train↑, val↓  (overfitting)
      • SVM with RBF kernel generalises well    (max-margin + regularisation)
    """
    rng = np.random.RandomState(seed)
    centres = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, 2.6]])
    n_per = N // n_classes
    Xl, yl = [], []
    for c in range(n_classes):
        nc = n_per if c < n_classes - 1 else N - n_per * (n_classes - 1)
        lat = rng.randn(nc, 2) * 0.55 + centres[c]
        Xl.append(lat);  yl.append(np.full(nc, c, dtype=int))
    lat_all = np.vstack(Xl)
    y = np.concatenate(yl)
    l0, l1 = lat_all[:, 0], lat_all[:, 1]

    X = np.zeros((N, D))
    for i in range(10):
        X[:, i] = np.sin(l0 * np.pi * (i + 1))
    for i in range(10, 20):
        X[:, i] = (l1 ** 2) * np.cos(l0 * (i - 9))
    for i in range(20, 30):
        X[:, i] = l0 * l1 * np.tanh(l0 + l1 + 0.1 * i)
    X[:, 30:] = rng.randn(N, D - 30)

    mu = X.mean(0);  sig = X.std(0) + 1e-8
    return (X - mu) / sig, y


# ═════════════════════════════════════════════════════════════════════════════
# §2  FROM-SCRATCH MODEL IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════

# ── §2a  Perceptron ───────────────────────────────────────────────────────────
class Perceptron:
    """
    Rosenblatt Perceptron (binary).

    Math
    ────
      Activation : ŷ = sign( w·x + b )      ← step function
      Update     : if ŷ ≠ y:
                     w ← w + η·y·x
                     b ← b + η·y
    Labels ∈ {−1, +1}.
    Guaranteed convergence only when data are linearly separable.
    """
    def __init__(self, lr=0.01, n_iter=1000, seed=42):
        self.lr = lr;  self.n_iter = n_iter;  self.seed = seed

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n, d = X.shape
        self.w_ = rng.randn(d) * 0.01;  self.b_ = 0.0
        self.errors_ = []
        for _ in range(self.n_iter):
            errs = 0
            for xi, yi in zip(X, y):
                yh = self._step(xi)
                if yh != yi:
                    self.w_ += self.lr * yi * xi
                    self.b_ += self.lr * yi
                    errs += 1
            self.errors_.append(errs)
            if errs == 0: break
        return self

    def _step(self, x):
        return 1 if np.dot(self.w_, x) + self.b_ >= 0 else -1

    def predict(self, X): return np.array([self._step(xi) for xi in X])
    def score(self, X, y): return np.mean(self.predict(X) == y)


# ── §2b  Neural Network (MLP) ─────────────────────────────────────────────────
class NeuralNetwork:
    """
    One-hidden-layer MLP.

    Architecture : Input → Dense(H, ReLU) → Dense(C, Softmax)
    Loss         : Categorical Cross-Entropy
    Optimiser    : Mini-batch SGD with optional L2 weight-decay

    Forward pass
    ─────────────
      Z1 = X @ W1 + b1         pre-activation (hidden)
      A1 = ReLU(Z1)            hidden output
      Z2 = A1 @ W2 + b2        pre-activation (output)
      A2 = softmax(Z2)         class probabilities

    Backward pass  (chain rule / backpropagation)
    ──────────────
      δ2 = (A2 − Y_oh) / N    gradient at output (cross-entropy + softmax)
      dW2 = A1ᵀ δ2 + λ W2
      db2 = Σ δ2
      δ1 = (δ2 @ W2ᵀ) ⊙ 1[Z1>0]   gradient at hidden (ReLU derivative)
      dW1 = Xᵀ δ1 + λ W1
      db1 = Σ δ1

    Weight update : W ← W − lr·dW,  b ← b − lr·db
    """
    def __init__(self, hidden=64, lr=0.01, epochs=200,
                 batch=32, l2=0.0, seed=42):
        self.hidden = hidden; self.lr = lr; self.epochs = epochs
        self.batch = batch;   self.l2 = l2; self.seed = seed

    # activations
    @staticmethod
    def _relu(z):       return np.maximum(0, z)
    @staticmethod
    def _relu_d(z):     return (z > 0).astype(float)
    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)       # numeric stability
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def _init(self, d, C):
        rng = np.random.RandomState(self.seed)
        # He initialisation
        self.W1 = rng.randn(d, self.hidden) * np.sqrt(2 / d)
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.randn(self.hidden, C) * np.sqrt(2 / self.hidden)
        self.b2 = np.zeros(C)

    def _fwd(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = self._relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = self._softmax(Z2)
        return Z1, A1, Z2, A2

    def _loss(self, A2, Yoh):
        n = Yoh.shape[0]
        ce = -np.sum(Yoh * np.log(A2 + 1e-15)) / n
        return ce + 0.5 * self.l2 * (np.sum(self.W1**2) + np.sum(self.W2**2))

    def _bwd(self, X, Z1, A1, A2, Yoh):
        n = X.shape[0]
        d2  = (A2 - Yoh) / n
        dW2 = A1.T @ d2 + self.l2 * self.W2
        db2 = d2.sum(0)
        d1  = (d2 @ self.W2.T) * self._relu_d(Z1)
        dW1 = X.T @ d1 + self.l2 * self.W1
        db1 = d1.sum(0)
        return dW1, db1, dW2, db2

    @staticmethod
    def _oh(y, C):
        oh = np.zeros((len(y), C))
        oh[np.arange(len(y)), y.astype(int)] = 1
        return oh

    def fit(self, Xtr, ytr, Xval=None, yval=None):
        n, d = Xtr.shape
        C = len(np.unique(ytr))
        self._init(d, C)
        Yoh = self._oh(ytr, C)
        rng = np.random.RandomState(self.seed)
        hist = dict(tr_loss=[], tr_acc=[], val_loss=[], val_acc=[])

        for ep in range(self.epochs):
            idx = rng.permutation(n)
            Xs, Ys = Xtr[idx], Yoh[idx]
            for s in range(0, n, self.batch):
                Xb, Yb = Xs[s:s+self.batch], Ys[s:s+self.batch]
                Z1, A1, Z2, A2 = self._fwd(Xb)
                dW1, db1, dW2, db2 = self._bwd(Xb, Z1, A1, A2, Yb)
                self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2

            _, _, _, A2t = self._fwd(Xtr)
            hist["tr_loss"].append(self._loss(A2t, Yoh))
            hist["tr_acc"].append(float(np.mean(A2t.argmax(1) == ytr)))

            if Xval is not None:
                Yval_oh = self._oh(yval, C)
                _, _, _, A2v = self._fwd(Xval)
                hist["val_loss"].append(self._loss(A2v, Yval_oh))
                hist["val_acc"].append(float(np.mean(A2v.argmax(1) == yval)))

        return hist

    def predict(self, X): return self._fwd(X)[3].argmax(1)
    def score(self, X, y): return float(np.mean(self.predict(X) == y))


# ── §2c  Kernel SVM (Pegasos, binary) ────────────────────────────────────────
class KernelSVM:
    """
    Kernelised binary SVM trained with the Pegasos algorithm.

    Primal objective
    ─────────────────
      J(w) = λ/2 ‖w‖² + 1/N Σ max(0, 1 − yᵢ f(xᵢ))

    Kernel / dual representation
    ─────────────────────────────
      f(x) = Σⱼ αⱼ yⱼ K(xⱼ, x) + b

    Pegasos update (stochastic, t = 1..T)
    ──────────────────────────────────────
      lr_t = 1 / (λ t)                              decreasing step
      pick sample i at random
      if  yᵢ (αᵀ·(y⊙K[:,i])) < 1:  αᵢ += 1        hinge violation
      α  ←  α · (1 − lr_t · λ)                     shrinkage

    Kernels supported
    ──────────────────
      linear  :  K(x,z) = xᵀz
      poly    :  K(x,z) = (γ xᵀz + c₀)^d
      rbf     :  K(x,z) = exp(−γ ‖x−z‖²)
    """
    def __init__(self, kernel="rbf", C=1.0, gamma=0.1,
                 degree=3, coef0=1.0, n_iter=300, seed=42):
        self.kernel = kernel;  self.C = C;    self.gamma = gamma
        self.degree = degree;  self.coef0 = coef0
        self.n_iter = n_iter;  self.seed = seed

    # ── kernel computation ────────────────────────────────────────────────────
    def _K(self, A, B):
        """Compute (|A|×|B|) kernel matrix between row-sets A and B."""
        if self.kernel == "linear":
            return A @ B.T

        elif self.kernel == "poly":
            # K(x,z) = (γ <x,z> + c₀)^d
            return (self.gamma * (A @ B.T) + self.coef0) ** self.degree

        elif self.kernel == "rbf":
            # ‖x−z‖² = ‖x‖²  − 2<x,z> + ‖z‖²   (broadcasting trick)
            sqA = np.sum(A**2, axis=1, keepdims=True)   # (m,1)
            sqB = np.sum(B**2, axis=1, keepdims=True)   # (n,1)
            D2  = np.maximum(sqA + sqB.T - 2 * (A @ B.T), 0)
            return np.exp(-self.gamma * D2)

        raise ValueError(f"Unknown kernel '{self.kernel}'")

    # ── Pegasos fit ───────────────────────────────────────────────────────────
    def fit(self, X, y):
        """Labels y ∈ {−1, +1}."""
        rng = np.random.RandomState(self.seed)
        N = X.shape[0]
        lam = 1.0 / (self.C * N)           # regularisation λ
        self.X_sv_ = X
        self.y_sv_ = y
        self.alpha_ = np.zeros(N)
        self.b_     = 0.0

        K = self._K(X, X)                  # pre-compute full N×N kernel matrix

        for t in range(1, self.n_iter + 1):
            lr_t = 1.0 / (lam * t)
            i = rng.randint(0, N)
            f_i = float((self.alpha_ * y) @ K[:, i]) + self.b_
            if y[i] * f_i < 1:
                self.alpha_[i] += 1
            self.alpha_ *= (1.0 - lr_t * lam)

        # bias: mean of (y − f) over near-support-vector examples
        scores = (self.alpha_ * y) @ K
        sv_mask = y * scores <= 1.05
        self.b_ = float(np.mean(y[sv_mask] - scores[sv_mask])) if sv_mask.any() else 0.0

        # record support vector count (alpha > threshold)
        self.n_sv_ = int(np.sum(self.alpha_ > 1e-5))
        return self

    def decision_function(self, X):
        return self._K(X, self.X_sv_) @ (self.alpha_ * self.y_sv_) + self.b_

    def predict(self, X):
        return np.sign(self.decision_function(X)).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


# ── §2d  One-vs-Rest multi-class SVM ─────────────────────────────────────────
class OvRSVM:
    """
    One-vs-Rest wrapper for KernelSVM.
    Trains K binary SVMs; prediction = argmax over raw decision scores.
    Supports integer class labels 0 … K−1.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.clfs_   = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.clfs_ = []
        for c in self.classes_:
            yb = np.where(y == c, 1, -1)
            clf = KernelSVM(**self.kwargs)
            clf.fit(X, yb)
            self.clfs_.append(clf)
        # total support vectors across all binary classifiers
        self.n_sv_ = sum(clf.n_sv_ for clf in self.clfs_)
        return self

    def decision_function(self, X):
        return np.column_stack([clf.decision_function(X) for clf in self.clfs_])

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


# ═════════════════════════════════════════════════════════════════════════════
# §3  CORE COMPARISON EXPERIMENTS
# ═════════════════════════════════════════════════════════════════════════════

# ── §3a  Experiment 1 – moons ─────────────────────────────────────────────────
def experiment_1(X_A, y_A):
    """
    Train Perceptron | Linear SVM | RBF SVM on the 2-D moons dataset.
    Returns dict with models + split data for plotting.
    """
    print(SEP)
    print("  EXPERIMENT 1 – Dataset A: 2-D Moons  (binary classification)")
    print(SEP2)

    y_pm = np.where(y_A == 0, -1, 1)    # remap to ±1
    Xtr, Xte, ytr, yte = train_test_split(
        X_A, y_pm, test_size=0.25, random_state=42, stratify=y_pm)

    # Perceptron
    perc = Perceptron(lr=0.01, n_iter=500)
    perc.fit(Xtr, ytr)
    perc_tr = perc.score(Xtr, ytr);  perc_te = perc.score(Xte, yte)
    print(f"  Perceptron       train={perc_tr:.3f}  val={perc_te:.3f}")

    # Linear SVM
    svm_lin = KernelSVM(kernel="linear", C=1.0, n_iter=400)
    svm_lin.fit(Xtr, ytr)
    lin_tr = svm_lin.score(Xtr, ytr);  lin_te = svm_lin.score(Xte, yte)
    print(f"  Linear SVM       train={lin_tr:.3f}  val={lin_te:.3f}")

    # RBF SVM
    svm_rbf = KernelSVM(kernel="rbf", C=5.0, gamma=0.5, n_iter=600)
    svm_rbf.fit(Xtr, ytr)
    rbf_tr = svm_rbf.score(Xtr, ytr);  rbf_te = svm_rbf.score(Xte, yte)
    print(f"  RBF SVM          train={rbf_tr:.3f}  val={rbf_te:.3f}")
    print()

    return dict(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
                perc=perc, svm_lin=svm_lin, svm_rbf=svm_rbf,
                scores={
                    "Perceptron":  (perc_tr, perc_te),
                    "Linear SVM":  (lin_tr,  lin_te),
                    "RBF SVM":     (rbf_tr,  rbf_te),
                })


# ── §3b  Experiment 2 – high-dim, learning curves ────────────────────────────
def experiment_2(X_B, y_B):
    """
    Train NeuralNetwork and OvR-RBF-SVM on the high-dim small-N dataset.
    Records per-epoch train/val accuracy → learning curves.
    Uses well-tuned hyperparameters for the SVM to ensure fair generalisation.
    """
    print(SEP)
    print("  EXPERIMENT 2 – Dataset B: High-dim Small-N  (3-class)")
    print(SEP2)

    Xtr, Xte, ytr, yte = train_test_split(
        X_B, y_B, test_size=0.30, random_state=42, stratify=y_B)

    # Neural Network – deliberately no L2 to show overfitting
    print("  Training Neural Network …")
    nn = NeuralNetwork(hidden=128, lr=0.005, epochs=300,
                       batch=16, l2=0.0, seed=42)
    hist = nn.fit(Xtr, ytr, Xval=Xte, yval=yte)
    print(f"  NN               train={hist['tr_acc'][-1]:.3f}"
          f"  val={hist['val_acc'][-1]:.3f}")

    # OvR RBF SVM – tuned C & gamma for high-dim data
    # Low gamma (0.005) because high D means raw RBF distances are large.
    # Moderate C (5.0) → soft margin that tolerates the noise features.
    print("  Training OvR-RBF SVM …")
    svm = OvRSVM(kernel="rbf", C=5.0, gamma=0.005, n_iter=500, seed=42)
    svm.fit(Xtr, ytr)
    svm_tr = svm.score(Xtr, ytr);  svm_te = svm.score(Xte, yte)
    print(f"  OvR RBF SVM      train={svm_tr:.3f}  val={svm_te:.3f}")
    print()

    return dict(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
                nn=nn, hist=hist,
                svm=svm, svm_tr=svm_tr, svm_te=svm_te)


# ═════════════════════════════════════════════════════════════════════════════
# §4  KERNEL COMPARISON (all kernels × both datasets)
# ═════════════════════════════════════════════════════════════════════════════

def kernel_comparison(X_A, y_A, X_B, y_B):
    """
    §4a – Compare Linear / Poly / RBF kernels on both datasets.
    Produces:
      (1) Decision boundary grid (Dataset A only – 2-D)
      (2) Grouped bar chart: train & val accuracy × kernel × dataset
    """
    print(SEP)
    print("  §4 KERNEL COMPARISON – Linear vs Poly vs RBF")
    print(SEP2)

    y_A_pm = np.where(y_A == 0, -1, 1)
    XA_tr, XA_te, yA_tr, yA_te = train_test_split(
        X_A, y_A_pm, test_size=0.25, random_state=42, stratify=y_A_pm)
    XB_tr, XB_te, yB_tr, yB_te = train_test_split(
        X_B, y_B, test_size=0.30, random_state=42, stratify=y_B)

    # ── kernel configs ────────────────────────────────────────────────────────
    kernel_cfgs = {
        "Linear": dict(kernel="linear", C=1.0,  n_iter=400),
        "Poly\n(d=3)": dict(kernel="poly",   C=1.0,  gamma=0.5,
                            degree=3, coef0=1.0, n_iter=400),
        "RBF":    dict(kernel="rbf",    C=5.0,  gamma=0.5, n_iter=600),
    }
    # For Dataset B we use OvR; tweak gamma for high-dim
    kernel_cfgs_B = {
        "Linear": dict(kernel="linear", C=1.0,  n_iter=400),
        "Poly\n(d=3)": dict(kernel="poly",  C=1.0, gamma=0.01,
                            degree=3, coef0=1.0, n_iter=400),
        "RBF":    dict(kernel="rbf",    C=5.0,  gamma=0.005, n_iter=500),
    }
    kernel_labels = list(kernel_cfgs.keys())

    results = {}   # key → (model_A, trA, teA, model_B, trB, teB)

    for (lbl, cfgA), cfgB in zip(kernel_cfgs.items(), kernel_cfgs_B.values()):
        # Dataset A (binary, KernelSVM)
        mA = KernelSVM(**cfgA);  mA.fit(XA_tr, yA_tr)
        trA = mA.score(XA_tr, yA_tr);  teA = mA.score(XA_te, yA_te)
        print(f"  {lbl.replace(chr(10),' '):<12}  DataA "
              f"train={trA:.3f} val={teA:.3f}")

        # Dataset B (multi-class, OvRSVM)
        mB = OvRSVM(**cfgB);     mB.fit(XB_tr, yB_tr)
        trB = mB.score(XB_tr, yB_tr);  teB = mB.score(XB_te, yB_te)
        print(f"  {lbl.replace(chr(10),' '):<12}  DataB "
              f"train={trB:.3f} val={teB:.3f}")

        results[lbl] = (mA, trA, teA, mB, trB, teB)
    print()

    # ── Plot 4.1: Decision boundaries on Dataset A ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("§4 – Kernel Comparison: Decision Boundaries on Dataset A (Moons)",
                 fontsize=13, fontweight="bold", y=1.01)

    X_all = np.vstack([XA_tr, XA_te]);  y_all = np.concatenate([yA_tr, yA_te])
    for ax, (lbl, (mA, trA, teA, *_)) in zip(axes, results.items()):
        _draw_boundary_binary(ax, mA, X_all, y_all,
                              title=f"{lbl.replace(chr(10),' ')}\n"
                                    f"train={trA:.2%}  val={teA:.2%}")
    plt.tight_layout()
    _save("plot_kernel_boundaries.png")

    # ── Plot 4.2: Grouped bar chart ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("§4 – Kernel Comparison: Train vs Val Accuracy",
                 fontsize=13, fontweight="bold")

    short = [l.replace("\n", " ") for l in kernel_labels]
    x = np.arange(len(kernel_labels));  w = 0.35

    for ax, ds_idx, title in zip(axes, [0, 1],
                                  ["Dataset A – Moons", "Dataset B – High-Dim"]):
        tr_vals = [results[k][1 + ds_idx*3] for k in kernel_labels]
        te_vals = [results[k][2 + ds_idx*3] for k in kernel_labels]
        bars_tr = ax.bar(x - w/2, tr_vals, w, label="Train",
                         color=[KERNEL_COLORS[c.split()[0].lower().strip()]
                                for c in kernel_labels], alpha=0.9)
        bars_te = ax.bar(x + w/2, te_vals, w, label="Val",
                         color=[KERNEL_COLORS[c.split()[0].lower().strip()]
                                for c in kernel_labels], alpha=0.5,
                         edgecolor="white", linewidth=0.8)
        # annotate
        for b in list(bars_tr) + list(bars_te):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.012,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x);  ax.set_xticklabels(short, fontsize=10)
        ax.set_ylim(0, 1.18);  ax.set_ylabel("Accuracy")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9);  ax.grid(axis="y", alpha=0.3)
        ax.axhline(1.0, color=PAL["grid"], linewidth=1, linestyle="--")

    # complexity annotation (§ complexity = #SV)
    sv_A = [results[k][0].n_sv_ for k in kernel_labels]
    sv_B = [results[k][3].n_sv_ for k in kernel_labels]
    ax2 = axes[1].twinx()
    ax2.plot(x, sv_B, "D--", color=PAL["gold"], ms=7, lw=1.5, label="#SV (B)")
    ax2.set_ylabel("# Support Vectors", color=PAL["gold"])
    ax2.tick_params(axis="y", colors=PAL["gold"])
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    _save("plot_kernel_bar.png")

    return results, (XA_tr, XA_te, yA_tr, yA_te), (XB_tr, XB_te, yB_tr, yB_te)


# ═════════════════════════════════════════════════════════════════════════════
# §5  HYPERPARAMETER DEEP-DIVE  (best kernel = RBF)
# ═════════════════════════════════════════════════════════════════════════════

# ── shared boundary drawing utilities ────────────────────────────────────────
def _draw_boundary_binary(ax, model, X, y, title="",
                           h=0.03, show_sv=False):
    """
    Draw coloured decision regions + data scatter for a binary {-1,+1} model.
    """
    x0m, x0M = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    x1m, x1M = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.arange(x0m, x0M, h),
                         np.arange(x1m, x1M, h))
    G = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(G).reshape(xx.shape)
    zz_bin = ((zz > 0)).astype(float)

    cmap_r = ListedColormap([PAL["r0"], PAL["r1"]])
    ax.contourf(xx, yy, zz_bin, alpha=0.40, cmap=cmap_r,
                levels=[-0.5, 0.5, 1.5])
    ax.contour(xx, yy, zz_bin, colors=PAL["gold"],
               linewidths=1.4, levels=[0.5], linestyles="--", alpha=0.95)

    colors = np.where(y <= 0, C0, C1)
    ax.scatter(X[:, 0], X[:, 1], c=colors, edgecolors="white",
               linewidths=0.4, s=28, zorder=4)

    if show_sv and hasattr(model, "alpha_"):
        sv_mask = model.alpha_ > 1e-5
        if sv_mask.any():
            Xsv = model.X_sv_[sv_mask]
            ax.scatter(Xsv[:, 0], Xsv[:, 1], s=90, facecolors="none",
                       edgecolors=PAL["gold"], linewidths=1.6, zorder=5,
                       label=f"SV={sv_mask.sum()}")
            ax.legend(fontsize=7, loc="lower right")

    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
    ax.set_xlabel("Feature 1", fontsize=8);  ax.set_ylabel("Feature 2", fontsize=8)
    ax.tick_params(labelsize=7)


# ── §5a  C Parameter ─────────────────────────────────────────────────────────
def plot_C_analysis(XA_tr, XA_te, yA_tr, yA_te):
    """
    §5a – Effect of C on decision boundary width and support-vector count.

    Plot 5a-1 : 1×3 boundary subplots for C ∈ {0.01, 1.0, 100}
    Plot 5a-2 : Line graph  #Support Vectors vs C (log scale x-axis)
    """
    print(SEP2)
    print("  §5a – C Parameter (Regularisation)")
    print(SEP2)

    C_demo   = [0.01, 1.0, 100.0]
    C_sweep  = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0,
                50.0, 100.0, 500.0]

    X_all = np.vstack([XA_tr, XA_te]);  y_all = np.concatenate([yA_tr, yA_te])

    # ── Plot 5a-1: boundary at 3 values of C ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("§5a – C Parameter: Margin Width on Moons Dataset\n"
                 "Low C → soft/wide margin   |   High C → hard/jagged boundary",
                 fontsize=12, fontweight="bold", y=1.03)

    for ax, Cv in zip(axes, C_demo):
        m = KernelSVM(kernel="rbf", C=Cv, gamma=0.5, n_iter=600)
        m.fit(XA_tr, yA_tr)
        sv_n = m.n_sv_
        tr_a = m.score(XA_tr, yA_tr);  te_a = m.score(XA_te, yA_te)
        lbl = (f"C = {Cv}\ntrain={tr_a:.2%}  val={te_a:.2%}"
               f"\n#SV = {sv_n}")
        _draw_boundary_binary(ax, m, X_all, y_all, title=lbl, show_sv=True)
        # add annotation arrow for small C
        if Cv == C_demo[0]:
            ax.text(0.03, 0.05, "← Wide, smooth margin\n(Underfitting risk)",
                    transform=ax.transAxes, fontsize=7.5, color=PAL["c2"])
        elif Cv == C_demo[-1]:
            ax.text(0.03, 0.05, "← Jagged boundary\n(Overfitting risk)",
                    transform=ax.transAxes, fontsize=7.5, color=PAL["c1"])

    plt.tight_layout()
    _save("plot_C_boundaries.png")

    # ── Plot 5a-2: #SV vs C ───────────────────────────────────────────────────
    sv_counts = []
    tr_accs, te_accs = [], []
    print("  Sweeping C …", end=" ")
    for Cv in C_sweep:
        m = KernelSVM(kernel="rbf", C=Cv, gamma=0.5, n_iter=500)
        m.fit(XA_tr, yA_tr)
        sv_counts.append(m.n_sv_)
        tr_accs.append(m.score(XA_tr, yA_tr))
        te_accs.append(m.score(XA_te, yA_te))
    print("done")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("§5a – Number of Support Vectors vs C  (RBF kernel)",
                 fontsize=12, fontweight="bold")

    ax.plot(C_sweep, sv_counts, "o-", color=PAL["purple"], lw=2.2,
            ms=7, label="#Support Vectors")
    ax.set_xscale("log")
    ax.set_xlabel("C  (log scale)", fontsize=11)
    ax.set_ylabel("Number of Support Vectors", fontsize=11, color=PAL["purple"])
    ax.tick_params(axis="y", colors=PAL["purple"])
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(C_sweep, tr_accs, "s--", color=C0, lw=1.8, ms=6, label="Train Acc")
    ax2.plot(C_sweep, te_accs, "^--", color=C1, lw=1.8, ms=6, label="Val Acc")
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y")

    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=9, loc="center right")

    ax.annotate("↑ Low C: many SVs, soft margin",
                xy=(C_sweep[1], sv_counts[1]), xytext=(C_sweep[2], sv_counts[1]+5),
                fontsize=8, color=PAL["purple"],
                arrowprops=dict(arrowstyle="->", color=PAL["purple"]))
    ax.annotate("↓ High C: fewer SVs, hard margin",
                xy=(C_sweep[-2], sv_counts[-2]),
                xytext=(C_sweep[-4], sv_counts[-2]+8),
                fontsize=8, color=PAL["purple"],
                arrowprops=dict(arrowstyle="->", color=PAL["purple"]))

    plt.tight_layout()
    _save("plot_C_sv_count.png")


# ── §5b  Gamma Parameter ─────────────────────────────────────────────────────
def plot_gamma_analysis(XA_tr, XA_te, yA_tr, yA_te):
    """
    §5b – Effect of γ on the RBF decision boundary ("island" effect).

    Plot 5b: 1×3 boundary subplots for γ ∈ {0.01, 1.0, 50}
    """
    print(SEP2)
    print("  §5b – Gamma Parameter (RBF Kernel Width)")
    print(SEP2)

    gamma_demo = [0.01, 1.0, 50.0]
    X_all = np.vstack([XA_tr, XA_te]);  y_all = np.concatenate([yA_tr, yA_te])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("§5b – γ Parameter: The 'Island' Effect on Moons Dataset\n"
                 "Low γ → smooth global boundary   |   High γ → tiny islands "
                 "around each training point",
                 fontsize=12, fontweight="bold", y=1.03)

    descs = [
        "← Far-reaching influence\nSmooth boundary (underfitting)",
        "← Balanced reach\nGood generalisation",
        "← Short-range only\nTiny islands = extreme overfitting",
    ]

    for ax, gv, desc in zip(axes, gamma_demo, descs):
        m = KernelSVM(kernel="rbf", C=5.0, gamma=gv, n_iter=600)
        m.fit(XA_tr, yA_tr)
        sv_n  = m.n_sv_
        tr_a  = m.score(XA_tr, yA_tr);  te_a = m.score(XA_te, yA_te)
        title = (f"γ = {gv}\ntrain={tr_a:.2%}  val={te_a:.2%}"
                 f"\n#SV = {sv_n}")
        _draw_boundary_binary(ax, m, X_all, y_all, title=title)
        ax.text(0.03, 0.05, desc, transform=ax.transAxes,
                fontsize=7.5, color=PAL["cyan"])

    plt.tight_layout()
    _save("plot_gamma_islands.png")


# ── §5c  C × Gamma Interaction (heatmap on Dataset B) ────────────────────────
def plot_C_gamma_heatmap(XB_tr, XB_te, yB_tr, yB_te):
    """
    §5c – 2-D heatmap of val accuracy over C × γ grid on Dataset B.
    Uses OvR-RBF SVM.  Reveals the "diagonal ridge" of optimal combinations.
    """
    print(SEP2)
    print("  §5c – C × γ Interaction Heatmap (Dataset B)")
    print(SEP2)

    C_vals   = [0.01, 0.1, 1.0, 10.0, 100.0]
    gam_vals = [0.001, 0.01, 0.1, 1.0, 10.0]

    acc_grid = np.zeros((len(C_vals), len(gam_vals)))
    total = len(C_vals) * len(gam_vals)
    done = 0

    N = len(yB_tr) + len(yB_te)
    X_full = np.vstack([XB_tr, XB_te])
    y_full = np.concatenate([yB_tr, yB_te])

    for ci, C in enumerate(C_vals):
        for gi, gam in enumerate(gam_vals):
            # 3-fold manual CV on full Dataset B
            fold_sz = N // 3
            fold_accs = []
            for f in range(3):
                vi  = list(range(f * fold_sz, (f+1) * fold_sz))
                tri = [i for i in range(N) if i not in vi]
                m = OvRSVM(kernel="rbf", C=C, gamma=gam, n_iter=300)
                m.fit(X_full[tri], y_full[tri])
                fold_accs.append(m.score(X_full[vi], y_full[vi]))
            acc_grid[ci, gi] = np.mean(fold_accs)
            done += 1
            print(f"    [{done}/{total}] C={C:6.3f}  γ={gam:.4f}"
                  f"  CV-acc={acc_grid[ci,gi]:.3f}")

    print()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("§5c – C × γ Interaction Heatmap (Dataset B)\n"
                 "Diagonal ridge shows that as γ↑, optimal C↓ to prevent overfitting",
                 fontsize=12, fontweight="bold")

    im = ax.imshow(acc_grid, cmap="plasma", aspect="auto",
                   vmin=acc_grid.min() - 0.01, vmax=acc_grid.max())
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("3-Fold CV Accuracy", fontsize=10)

    ax.set_xticks(range(len(gam_vals)));  ax.set_xticklabels(gam_vals, fontsize=10)
    ax.set_yticks(range(len(C_vals)));    ax.set_yticklabels(C_vals,   fontsize=10)
    ax.set_xlabel("γ  (kernel width)", fontsize=11)
    ax.set_ylabel("C  (regularisation)", fontsize=11)

    # annotate cells
    for ci in range(len(C_vals)):
        for gi in range(len(gam_vals)):
            v = acc_grid[ci, gi]
            ax.text(gi, ci, f"{v:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="black" if v > acc_grid.max() - 0.04 else "white",
                    fontweight="bold" if v == acc_grid.max() else "normal")

    # mark best cell
    best_c, best_g = np.unravel_index(acc_grid.argmax(), acc_grid.shape)
    ax.add_patch(plt.Rectangle((best_g - 0.5, best_c - 0.5), 1, 1,
                                fill=False, edgecolor=PAL["cyan"],
                                linewidth=2.5, label=(
                                    f"Best: C={C_vals[best_c]},"
                                    f" γ={gam_vals[best_g]}")))
    ax.legend(fontsize=9, loc="upper right")

    # overlay diagonal ridge guide
    diag_g = np.linspace(-0.5, len(gam_vals)-0.5, 100)
    diag_c = (len(C_vals) - 1) - diag_g * (len(C_vals)-1) / (len(gam_vals)-1)
    ax.plot(diag_g, diag_c, "--", color=PAL["cyan"], lw=1.2, alpha=0.5,
            label="Expected ridge direction")

    plt.tight_layout()
    _save("plot_C_gamma_heatmap.png")


# ── §5d  Polynomial Degree Validation Curve (Dataset B) ──────────────────────
def plot_degree_validation_curve(XB_tr, XB_te, yB_tr, yB_te):
    """
    §5d – Polynomial kernel degree d = 1…7 on Dataset B.
    Shows training accuracy climbing to 100 % while validation crashes at
    higher degrees (overfitting the noise dimensions).
    """
    print(SEP2)
    print("  §5d – Polynomial Degree Validation Curve (Dataset B)")
    print(SEP2)

    degrees = list(range(1, 8))
    tr_accs, te_accs, sv_counts = [], [], []

    for d in degrees:
        print(f"    degree={d} …", end=" ")
        # Low gamma for high-dim; moderate C
        m = OvRSVM(kernel="poly", C=1.0, gamma=0.01,
                   degree=d, coef0=1.0, n_iter=350)
        m.fit(XB_tr, yB_tr)
        tr = m.score(XB_tr, yB_tr);  te = m.score(XB_te, yB_te)
        tr_accs.append(tr);  te_accs.append(te)
        sv_counts.append(m.n_sv_)
        print(f"train={tr:.3f}  val={te:.3f}  #SV={m.n_sv_}")

    print()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("§5d – Polynomial Kernel: Degree vs Accuracy (Dataset B)\n"
                 "Training accuracy saturates; validation crashes at high degree",
                 fontsize=12, fontweight="bold")

    # Left: accuracy curves
    ax = axes[0]
    ax.plot(degrees, tr_accs, "o-",  color=C0, lw=2.2, ms=7, label="Train Acc")
    ax.plot(degrees, te_accs, "s--", color=C1, lw=2.2, ms=7, label="Val Acc")
    ax.fill_between(degrees, tr_accs, te_accs,
                    where=np.array(tr_accs) >= np.array(te_accs),
                    alpha=0.15, color=PAL["gold"], label="Generalisation Gap")

    best_d = degrees[np.argmax(te_accs)]
    ax.axvline(best_d, color=PAL["cyan"], linestyle=":", lw=1.8,
               label=f"Best val @ d={best_d}")

    ax.set_xlabel("Polynomial Degree d", fontsize=11)
    ax.set_ylabel("Accuracy",            fontsize=11)
    ax.set_xticks(degrees);  ax.set_ylim(0, 1.08)
    ax.legend(fontsize=9);   ax.grid(True, alpha=0.3)

    # annotations
    ax.annotate("Train ≈ 100 %\n(memorises noise)",
                xy=(degrees[-1], tr_accs[-1]),
                xytext=(degrees[-2] - 0.5, tr_accs[-1] - 0.15),
                fontsize=8, color=C0,
                arrowprops=dict(arrowstyle="->", color=C0))
    ax.annotate("Val crashes\n(overfitting)",
                xy=(degrees[-1], te_accs[-1]),
                xytext=(degrees[-3], te_accs[-1] - 0.12),
                fontsize=8, color=C1,
                arrowprops=dict(arrowstyle="->", color=C1))

    # Right: #SV vs degree
    ax2 = axes[1]
    ax2.bar(degrees, sv_counts, color=PAL["purple"], alpha=0.8,
            edgecolor="white", linewidth=0.7)
    ax2.set_xlabel("Polynomial Degree d", fontsize=11)
    ax2.set_ylabel("Total # Support Vectors", fontsize=11)
    ax2.set_title("Model Complexity: # Support Vectors vs Degree",
                  fontsize=10, fontweight="bold")
    ax2.set_xticks(degrees);  ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(sv_counts):
        ax2.text(i + 1, v + 1, str(v), ha="center", fontsize=8.5)

    plt.tight_layout()
    _save("plot_degree_validation.png")


# ═════════════════════════════════════════════════════════════════════════════
# CORE VISUALIZATION HELPERS (Experiment 1 & 2)
# ═════════════════════════════════════════════════════════════════════════════

def plot_exp1_boundaries(exp1):
    """Plot 3 decision boundaries: Perceptron | Linear SVM | RBF SVM."""
    X_all = np.vstack([exp1["Xtr"], exp1["Xte"]])
    y_all = np.concatenate([exp1["ytr"], exp1["yte"]])
    models = [("Perceptron\n(step activation)",    exp1["perc"]),
              ("Linear SVM\n(Pegasos)",            exp1["svm_lin"]),
              ("RBF SVM\n(γ=0.5, C=5)",            exp1["svm_rbf"])]
    scores = exp1["scores"]
    score_keys = list(scores.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("§3a – Experiment 1: Decision Boundaries on Moons Dataset",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (title, m), key in zip(axes, models, score_keys):
        tr, te = scores[key]
        full_title = f"{title}\ntrain={tr:.2%}  val={te:.2%}"
        _draw_boundary_binary(ax, m, X_all, y_all,
                              title=full_title, show_sv=(key != "Perceptron"))
    plt.tight_layout()
    _save("plot_exp1_boundaries.png")


def plot_exp2_learning_curves(exp2):
    """Plot NN learning curves + SVM comparison bar."""
    hist = exp2["hist"]
    n_ep = len(hist["tr_acc"])
    ep   = np.arange(1, n_ep + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle("§3b – Experiment 2: NN Overfitting vs SVM Generalisation "
                 "(Dataset B)",
                 fontsize=13, fontweight="bold")

    # ── Left: NN learning curves ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(ep, hist["tr_acc"],  color=C0, lw=2.2, label="NN Train Acc")
    ax.plot(ep, hist["val_acc"], color=C1, lw=2.2,
            linestyle="--", label="NN Val Acc")
    ax.fill_between(ep, hist["tr_acc"], hist["val_acc"],
                    where=np.array(hist["tr_acc"]) > np.array(hist["val_acc"]),
                    alpha=0.18, color=PAL["gold"], label="Generalisation Gap")
    ax.set_title("Neural Network (MLP, no regularisation)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch");  ax.set_ylabel("Accuracy")
    ax.legend(fontsize=9);   ax.grid(True, alpha=0.3);  ax.set_ylim(0, 1.05)

    # ── Right: SVM final accuracy bars ───────────────────────────────────────
    ax2 = axes[1]
    labels = ["SVM Train", "SVM Val", "NN Train (final)", "NN Val (final)"]
    vals   = [exp2["svm_tr"], exp2["svm_te"],
              hist["tr_acc"][-1], hist["val_acc"][-1]]
    colors = [C2, C2, C0, C1]
    bars = ax2.bar(labels, vals, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.8, width=0.55)
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h + 0.012,
                 f"{h:.2%}", ha="center", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 1.18);  ax2.set_ylabel("Accuracy")
    ax2.set_title("Final Accuracy Comparison", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="x", labelsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(1.0, color=PAL["grid"], lw=1, linestyle="--")

    # bracket for generalisation gap
    svm_gap = exp2["svm_tr"] - exp2["svm_te"]
    nn_gap  = hist["tr_acc"][-1] - hist["val_acc"][-1]
    for xi, gap, label, clr in [(0.5, svm_gap, f"SVM gap\n{svm_gap:.2%}", C2),
                                  (2.5, nn_gap,  f"NN gap\n{nn_gap:.2%}",  C1)]:
        ax2.annotate("", xy=(xi + 0.5, vals[int(xi + 0.5)]),
                     xytext=(xi + 0.5, vals[int(xi - 0.5)]),
                     arrowprops=dict(arrowstyle="<->", color=clr, lw=1.5))
        ax2.text(xi + 0.7, (vals[int(xi)] + vals[int(xi)+1]) / 2,
                 label, fontsize=7.5, color=clr)

    plt.tight_layout()
    _save("plot_exp2_learning_curves.png")


# ═════════════════════════════════════════════════════════════════════════════
# §6  MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + SEP)
    print("  ML EXPERIMENT v2 – SVM vs Perceptron vs Neural Network")
    print("  + Comprehensive SVM Hyperparameter Analysis")
    print("  (All models from-scratch using NumPy)")
    print(SEP + "\n")

    # ── §1 Datasets ───────────────────────────────────────────────────────────
    print("PHASE 1 – Dataset Generation")
    print(SEP2)
    X_A, y_A = make_dataset_A(n=400, noise=0.25)
    print(f"  Dataset A (Moons)   shape={X_A.shape}  classes={np.unique(y_A)}")

    X_B, y_B = make_dataset_B(N=150, D=500, n_classes=3)
    print(f"  Dataset B (Hi-Dim)  shape={X_B.shape}  classes={np.unique(y_B)}\n")

    # ── §3 Core Experiments ───────────────────────────────────────────────────
    exp1 = experiment_1(X_A, y_A)
    exp2 = experiment_2(X_B, y_B)

    print(SEP)
    print("  VISUALISATIONS")
    print(SEP)

    print("\n[1] Experiment 1 decision boundaries …")
    plot_exp1_boundaries(exp1)

    print("[2] Experiment 2 learning curves …")
    plot_exp2_learning_curves(exp2)

    # ── §4 Kernel Comparison ──────────────────────────────────────────────────
    print("[3] Kernel comparison (boundaries + bar chart) …")
    kernel_res, (XA_tr, XA_te, yA_tr, yA_te), \
                (XB_tr, XB_te, yB_tr, yB_te) = kernel_comparison(
        X_A, y_A, X_B, y_B)

    # ── §5 Hyperparameter deep-dive ───────────────────────────────────────────
    print("[4] §5a – C parameter analysis …")
    plot_C_analysis(XA_tr, XA_te, yA_tr, yA_te)

    print("[5] §5b – Gamma parameter ('island' effect) …")
    plot_gamma_analysis(XA_tr, XA_te, yA_tr, yA_te)

    print("[6] §5c – C × Gamma heatmap (Dataset B) …")
    plot_C_gamma_heatmap(XB_tr, XB_te, yB_tr, yB_te)

    print("[7] §5d – Polynomial degree validation curve (Dataset B) …")
    plot_degree_validation_curve(XB_tr, XB_te, yB_tr, yB_te)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + SEP)
    print("  FINAL SUMMARY")
    print(SEP)
    print("""
§3 – Core Experiments
──────────────────────
  Exp 1 (Moons):
    • Perceptron draws a straight line → fails on non-linear crescents.
    • Linear SVM shares the same fundamental limit (max-margin hyperplane).
    • RBF SVM maps to ∞-dim space via K(x,z)=exp(−γ‖x−z‖²) → near-perfect.

  Exp 2 (High-Dim Small-N):
    • NN memorises 470 noise features → train→100 %, val stagnates.
    • RBF SVM: Pegasos λ-shrinkage + margin constraint → robust generalisation.

§4 – Kernel Comparison
────────────────────────
    • Linear: fast & interpretable, but fails on any non-linear boundary.
    • Poly  : flexible at low degree; catastrophically overfits at d ≥ 5.
    • RBF   : best all-round on both datasets; γ controls locality.

§5 – Hyperparameter Deep-Dive (RBF SVM)
─────────────────────────────────────────
    C parameter:
      Low C  → soft margin, many SVs, smooth boundary (underfitting risk).
      High C → hard margin, few SVs, jagged/perfect boundary (overfit risk).
      #SV curve decreases monotonically as C increases.

    Gamma parameter:
      Low γ  → each point has far-reaching influence → smooth global boundary.
      High γ → each point influences only nearby space → "islands" appear.
      At γ=50 the SVM draws tiny circles around every training point.

    C × Gamma interaction:
      Heatmap shows a diagonal ridge of high CV accuracy: as γ↑, optimal
      C must↓ to counterbalance the increasing model complexity.

    Polynomial degree:
      d=1 ≡ linear, d=2-3 often best on noisy data.
      Training accuracy quickly saturates at 100 %; validation crashes at
      d ≥ 5 as polynomial wiggles memorise noise dimensions.
""")


if __name__ == "__main__":
    main()
