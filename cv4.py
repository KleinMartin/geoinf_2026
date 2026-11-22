import math
import heapq
import random
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

K = 30


def loadPoints(file):
    P = []
    with open(file) as f:
        for line in f:
            x, y, z = line.split()
            P.append((float(x), float(y), float(z)))
    return P


def dist2(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2


def addCandidate(h, k, d2, j):
    if len(h) < k:
        heapq.heappush(h, (-d2, j))
    elif d2 < -h[0][0]:
        heapq.heapreplace(h, (-d2, j))


def worst(h, k):
    if len(h) < k:
        return math.inf
    return -h[0][0]


def heapToList(h):
    return sorted((-d, j) for d, j in h)


def boxDist2(p, box):
    d = 0
    for a in range(3):
        if p[a] < box[a]:
            d += (box[a] - p[a]) ** 2
        elif p[a] > box[a + 3]:
            d += (p[a] - box[a + 3]) ** 2
    return d


def boundingBox(P):
    xs, ys, zs = zip(*P)
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


class Naive:
    def __init__(self, P):
        self.P = P

    def knn(self, i, k):
        h = []
        for j in range(len(self.P)):
            if j != i:
                addCandidate(h, k, dist2(self.P[i], self.P[j]), j)
        return heapToList(h)


class Grid:
    def __init__(self, P, nr=None):
        self.P = P
        if nr is None:
            nr = round((len(P) / 0.4) ** (1 / 3))
        self.nr = nr
        self.box = boundingBox(P)
        self.size = [(self.box[a + 3] - self.box[a]) / nr for a in range(3)]

        self.H = defaultdict(list)
        self.J = []
        self.cells = []
        for i, p in enumerate(P):
            c = self.cellOf(p)
            j = c[0] + c[1] * nr + c[2] * nr * nr
            self.H[j].append(i)
            self.J.append(j)
            self.cells.append(c)

    def cellOf(self, p):
        c = []
        for a in range(3):
            t = int((p[a] - self.box[a]) / self.size[a])
            c.append(min(t, self.nr - 1))
        return c

    def knn(self, i, k):
        p = self.P[i]
        cx, cy, cz = self.cells[i]
        nr = self.nr
        h = []
        r = 0
        while True:
            for gx in range(cx - r, cx + r + 1):
                for gy in range(cy - r, cy + r + 1):
                    for gz in range(cz - r, cz + r + 1):
                        if max(abs(gx - cx), abs(gy - cy), abs(gz - cz)) != r:
                            continue
                        if not (0 <= gx < nr and 0 <= gy < nr and 0 <= gz < nr):
                            continue
                        for j in self.H.get(gx + gy * nr + gz * nr * nr, []):
                            if j != i:
                                addCandidate(h, k, dist2(p, self.P[j]), j)

            b = math.inf
            for a, c in enumerate((cx, cy, cz)):
                if c - r > 0:
                    b = min(b, p[a] - (self.box[a] + (c - r) * self.size[a]))
                if c + r < nr - 1:
                    b = min(b, self.box[a] + (c + r + 1) * self.size[a] - p[a])
            if b == math.inf:
                break
            if worst(h, k) <= b * b:
                break
            r += 1
        return heapToList(h)


class KDNode:
    def __init__(self, axis=None, split=None, left=None, right=None, points=None):
        self.axis = axis
        self.split = split
        self.left = left
        self.right = right
        self.points = points


class KDTree:
    def __init__(self, P, leaf=8):
        self.P = P
        self.leaf = leaf
        self.root = self.build(list(range(len(P))), 0)

    def build(self, idx, depth):
        if len(idx) <= self.leaf:
            return KDNode(points=idx)
        axis = depth % 3
        idx.sort(key=lambda i: self.P[i][axis])
        m = len(idx) // 2
        split = self.P[idx[m]][axis]
        return KDNode(axis, split, self.build(idx[:m], depth + 1), self.build(idx[m:], depth + 1))

    def knn(self, i, k):
        h = []
        self.search(self.root, i, k, h)
        return heapToList(h)

    def search(self, node, i, k, h):
        p = self.P[i]
        if node.points is not None:
            for j in node.points:
                if j != i:
                    addCandidate(h, k, dist2(p, self.P[j]), j)
            return
        diff = p[node.axis] - node.split
        if diff < 0:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left
        self.search(first, i, k, h)
        if diff * diff < worst(h, k):
            self.search(second, i, k, h)


class OctNode:
    def __init__(self, box):
        self.box = box
        self.children = []
        self.points = None


class Octree:
    def __init__(self, P, capacity=16, maxDepth=15):
        self.P = P
        self.capacity = capacity
        self.maxDepth = maxDepth
        b = boundingBox(P)
        s = max(b[3] - b[0], b[4] - b[1], b[5] - b[2]) * 1.0001
        self.root = self.build(list(range(len(P))), (b[0], b[1], b[2], b[0] + s, b[1] + s, b[2] + s), 0)

    def build(self, idx, box, depth):
        node = OctNode(box)
        if len(idx) <= self.capacity or depth == self.maxDepth:
            node.points = idx
            return node
        mid = [(box[a] + box[a + 3]) / 2 for a in range(3)]
        parts = [[] for _ in range(8)]
        for i in idx:
            p = self.P[i]
            code = 0
            for a in range(3):
                if p[a] >= mid[a]:
                    code += 2 ** a
            parts[code].append(i)
        for code in range(8):
            if not parts[code]:
                continue
            lo, hi = [], []
            for a in range(3):
                if code & (2 ** a):
                    lo.append(mid[a]); hi.append(box[a + 3])
                else:
                    lo.append(box[a]); hi.append(mid[a])
            node.children.append(self.build(parts[code], tuple(lo + hi), depth + 1))
        return node

    def knn(self, i, k):
        h = []
        self.search(self.root, i, k, h)
        return heapToList(h)

    def search(self, node, i, k, h):
        p = self.P[i]
        if node.points is not None:
            for j in node.points:
                if j != i:
                    addCandidate(h, k, dist2(p, self.P[j]), j)
            return
        children = sorted(node.children, key=lambda c: boxDist2(p, c.box))
        for c in children:
            if boxDist2(p, c.box) < worst(h, k):
                self.search(c, i, k, h)


class RNode:
    def __init__(self, leaf):
        self.leaf = leaf
        self.entries = []


def union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]),
            max(a[3], b[3]), max(a[4], b[4]), max(a[5], b[5]))


def nodeBox(node):
    b = node.entries[0][0]
    for e in node.entries[1:]:
        b = union(b, e[0])
    return b


class RTree:
    def __init__(self, P, M=16):
        self.P = P
        self.M = M
        self.m = 6
        b = boundingBox(P)
        self.eps = 1e-4 * max(b[3] - b[0], b[4] - b[1], b[5] - b[2])
        self.root = RNode(True)
        for i in range(len(P)):
            self.insert(i)

    def volume(self, b):
        e = self.eps
        return (b[3] - b[0] + e) * (b[4] - b[1] + e) * (b[5] - b[2] + e)

    def enlargement(self, b, new):
        return self.volume(union(b, new)) - self.volume(b)

    def insert(self, i):
        p = self.P[i]
        box = (p[0], p[1], p[2], p[0], p[1], p[2])
        node = self.root
        path = []
        while not node.leaf:
            best = min(node.entries, key=lambda e: (self.enlargement(e[0], box), self.volume(e[0])))
            path.append((node, best))
            node = best[1]
        node.entries.append([box, i])

        newNode = None
        if len(node.entries) > self.M:
            newNode = self.split(node)
        for parent, entry in reversed(path):
            entry[0] = nodeBox(entry[1])
            if newNode is not None:
                parent.entries.append([nodeBox(newNode), newNode])
                newNode = None
                if len(parent.entries) > self.M:
                    newNode = self.split(parent)
        if newNode is not None:
            old = self.root
            self.root = RNode(False)
            self.root.entries = [[nodeBox(old), old], [nodeBox(newNode), newNode]]

    def split(self, node):
        E = node.entries
        best = -1
        for a in range(len(E)):
            for b in range(a + 1, len(E)):
                d = self.volume(union(E[a][0], E[b][0])) - self.volume(E[a][0]) - self.volume(E[b][0])
                if d > best:
                    best, s1, s2 = d, a, b
        g1, g2 = [E[s1]], [E[s2]]
        b1, b2 = E[s1][0], E[s2][0]
        rest = [E[t] for t in range(len(E)) if t != s1 and t != s2]
        while rest:
            if len(g1) + len(rest) == self.m:
                g1 += rest
                break
            if len(g2) + len(rest) == self.m:
                g2 += rest
                break
            e = max(rest, key=lambda e: abs(self.enlargement(b1, e[0]) - self.enlargement(b2, e[0])))
            rest.remove(e)
            if self.enlargement(b1, e[0]) < self.enlargement(b2, e[0]):
                g1.append(e)
                b1 = union(b1, e[0])
            else:
                g2.append(e)
                b2 = union(b2, e[0])
        node.entries = g1
        newNode = RNode(node.leaf)
        newNode.entries = g2
        return newNode

    def knn(self, i, k):
        h = []
        self.search(self.root, i, k, h)
        return heapToList(h)

    def search(self, node, i, k, h):
        p = self.P[i]
        if node.leaf:
            for box, j in node.entries:
                if j != i:
                    addCandidate(h, k, dist2(p, self.P[j]), j)
            return
        entries = sorted(node.entries, key=lambda e: boxDist2(p, e[0]))
        for box, child in entries:
            if boxDist2(p, box) < worst(h, k):
                self.search(child, i, k, h)


class RTreeLib:
    def __init__(self, P):
        from rtree import index
        self.P = P
        prop = index.Property()
        prop.dimension = 3
        self.idx = index.Index(properties=prop)
        for i, p in enumerate(P):
            self.idx.insert(i, (p[0], p[1], p[2], p[0], p[1], p[2]))

    def knn(self, i, k):
        p = self.P[i]
        res = [(dist2(p, self.P[j]), j) for j in self.idx.nearest(p + p, k + 1) if j != i]
        return sorted(res)[:k]


def density(S, n):
    d = [math.sqrt(S.knn(i, 1)[0][0]) for i in range(n)]
    d_aver = sum(d) / n
    return 1 / d_aver ** 3, d_aver


def curvature(S, P):
    kappa = []
    for i in range(len(P)):
        idx = [i] + [j for d, j in S.knn(i, K)]
        Q = np.array([P[j] for j in idx])
        C = np.cov(Q.T)
        lam = np.linalg.eigvalsh(C)
        kappa.append(lam[0] / lam.sum())
    return np.array(kappa)


def measure(f):
    t = time.time()
    r = f()
    return r, time.time() - t


P = loadPoints('tree_18.txt')
n = len(P)
print('bodu:', n)

methods = {
    'voxelizace': Grid,
    'kd-tree': KDTree,
    'octree': Octree,
    'R-tree': RTree,
    'R-tree knihovna': RTreeLib,
}

random.seed(1)
sample = random.sample(P, 1000)
ref = Naive(sample)
for name, cls in methods.items():
    S = cls(sample)
    ok = all(np.allclose([d for d, j in S.knn(i, K)], [d for d, j in ref.knn(i, K)]) for i in range(len(sample)))
    print('kontrola', name, ok)

for name, cls in methods.items():
    S, tb = measure(lambda: cls(P))
    (rho, d_aver), td = measure(lambda: density(S, n))
    k, tk = measure(lambda: curvature(S, P))
    if name == 'kd-tree':
        kappa = k
    print(f'{name}: stavba {tb:.2f} s, hustota {td:.2f} s, krivost {tk:.2f} s, '
          f'd_aver = {d_aver:.4f} m, rho = {rho:.0f} bodu/m3')

A = np.array(P)
fig = plt.figure(figsize=(7, 9))
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(A[:, 0], A[:, 1], A[:, 2], c=kappa, cmap='jet', s=1,
                vmin=np.percentile(kappa, 1), vmax=np.percentile(kappa, 99))
ax.set_aspect('equal')
fig.colorbar(sc, shrink=0.5, label='krivost')
ax.set_title('Krivost (PCA, k = 30)')
plt.savefig('krivost.png', dpi=150)

sizes = [500, 1000, 2000, 4000, 8000, 16000, n]
times = defaultdict(list)
for m in sizes:
    Q = random.sample(P, m)
    if m <= 4000:
        times['naivni'].append(measure(lambda: density(Naive(Q), m))[1])
    for name, cls in methods.items():
        times[name].append(measure(lambda: density(cls(Q), m))[1])
    print('n =', m, 'hotovo')

plt.figure()
for name, t in times.items():
    plt.loglog(sizes[:len(t)], t, 'o-', label=name)
plt.xlabel('pocet bodu')
plt.ylabel('cas [s]')
plt.title('Hustota - cas vypoctu')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig('cas_n.png', dpi=150)

nrs = [3, 5, 8, 12, 16, 20, 30, 40, 55, 75]
vox, t_den, t_cur = [], [], []
for nr in nrs:
    G = Grid(P, nr)
    vox.append(np.mean(G.size))
    t_den.append(measure(lambda: density(G, n))[1])
    t_cur.append(measure(lambda: curvature(G, P))[1])
    print('nr =', nr, 'hotovo')

plt.figure()
plt.loglog(vox, t_den, 'o-', label='hustota')
plt.loglog(vox, t_cur, 's-', label='krivost')
plt.xlabel('velikost voxelu [m]')
plt.ylabel('cas [s]')
plt.title('Voxelizace - cas vs velikost voxelu')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig('cas_voxel.png', dpi=150)
plt.show()
