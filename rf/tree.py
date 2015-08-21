from string import Template
from copy import deepcopy

try:
	import pycuda.autoinit
	import pycuda.gpuarray as gpuarray
	import pycuda.driver as cuda
	from pycuda.compiler import SourceModule
except ImportError:
    pass

import numpy
import math

from .Splitter import SplitRecord


# ======================================================================
# Class Node
# ======================================================================
class Node:
	"""Class Node"""

	def __init__(self, parent, is_left, impurity,n_node_samples,stats,depth):
		""" init """
		self.parent = parent
		self.is_left = is_left
		self.impurity = impurity
		self.n_node_samples = n_node_samples
		self.stats = deepcopy(stats)
		self.depth = depth
	
	def isLeaf(self):
		"""isLeaf function"""
		return False

# ======================================================================
# Class Split
# ======================================================================
class Split(Node):
	"""Class Split"""
	 
	def __init__(self,parent, is_left, feature, threshold, impurity,
					n_node_samples, improvement, stats, depth):
		""" init """
		Node.__init__(self, parent, is_left, impurity, n_node_samples, stats, depth)
		self.feature = feature
		self.threshold = threshold
		self.childs=[-1,-1]
		self.improvement = improvement
		


	def __repr__(self):
		return "SPLIT %s, %s, %s, %s, %s, %s, %s " % (
					self.parent,
					self.impurity,
					self.n_node_samples,
					self.feature,
					self.threshold,
					self.childs,
					self.stats)

# ======================================================================
# Class Leaf
# ======================================================================
class Leaf(Node):
	"""Class Leaf"""
	 
	def __init__(self,parent, is_left, impurity, n_node_samples, stats, depth):
		""" init """
		Node.__init__(self, parent, is_left, impurity,n_node_samples, stats, depth)
		#self.stats = deepcopy(stats)

	def isLeaf(self):
		"""isLeaf function"""
		return True

	def __repr__(self):
		return "Leaf %s, %s, %s, %s " % (self.parent, self.impurity, self.n_node_samples, self.stats)

# ======================================================================
# class StackRecord
# ======================================================================
class StackRecord:
	"""Class StackRecord"""
	
	def __init__(self, start, end, depth, parent, is_left, impurity, stats):
		""" init """
		self.start = start
		self.end=end
		self.depth=depth
		self.parent=parent
		self.is_left=is_left
		self.impurity=impurity
		self.stats=deepcopy(stats)

	def __repr__(self):
		return "StackRecord %s, %s, %s, %s, %s, %s, %s " % (self.start, self.end, self.depth, self.parent, self.is_left, self.impurity, self.stats)

# ======================================================================
# Class Node for GPU
# ======================================================================
class Node_GPU:
	def __init__(self, ptr, isLeaf, threshold, proba,n_classes,
					idChildLeft, idChildRight, type,depth,
					c1,xm1,xM1,ym1,yM1,c2,xm2,xM2,ym2,yM2 ):

		shift = 0

		b = memoryview(numpy.uint32(isLeaf)); 		cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.float32(threshold)); 	cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.uint32(idChildLeft)); 	cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.uint32(idChildRight)); cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.uint32(depth)); 		cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.uint32(type)); 		cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.int32(c1)); 			cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.int32(xm1)); 			cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.int32(xM1)); 			cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.int32(ym1)); 			cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.int32(yM1)); 			cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.int32(c2)); 			cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.int32(xm2)); 			cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.int32(xM2)); 			cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.int32(ym2)); 			cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4
		b = memoryview(numpy.int32(yM2)); 			cuda.memcpy_htod(int(ptr)+shift, b); shift+= 4

		for i in range(n_classes):
				b = memoryview(numpy.float32(proba[i])); 		cuda.memcpy_htod(int(ptr)+shift, b); shift+= len(b)

		self.shift = shift

# ======================================================================
# Class Tree for GPU
# ======================================================================
class Tree_GPU:
	def __init__(self, tree, windows_x = 0, windows_y = 0, intermediate = False):

		node_size = 64+tree.n_classes*4
		self.ptr = cuda.mem_alloc(node_size*len(tree.nodes))
		shift=0
		dX = windows_x
		dY = windows_y

		for node in tree.nodes:
			isLeaf=False
			threshold=0.0
			idChildLeft =0
			idChildRight=0
			type = 0
			c1 = 0
			xm1 = 0
			xM1 = 0
			ym1 = 0
			yM1 = 0
			c2 = 0
			xm2 = 0
			xM2 = 0
			ym2 = 0
			yM2 = 0
			proba = [0]*tree.n_classes
			
			isLeaf = node.isLeaf() 
			if(intermediate and not isLeaf):
				isLeaf = (node.childs[0] == -1 and node.childs[1] == -1)
			else:
				isLeaf = node.isLeaf()
				
			if( isLeaf ):
				proba = node.stats
			else:
				idChildLeft  = node.childs[0]
				idChildRight = node.childs[1]
				threshold = node.threshold
				depth = node.depth
				param = node.feature.option['RQE']
				w = param['windows'][0]
				c1 = w['Channel']
				xm1 = w['Xmin'] - dX
				xM1 = w['Xmax'] - dX
				ym1 = w['Ymin'] - dY
				yM1 = w['Ymax'] - dY
				if param['type'] != "one":
					w = param['windows'][1]
					c2 = w['Channel']
					xm2 = w['Xmin'] - dX
					xM2 = w['Xmax'] - dX
					ym2 = w['Ymin'] - dY
					yM2 = w['Ymax'] - dY
					if param['type'] == 'sum':
						type = 1
					elif  param['type'] == 'diff':
						type = 2
					elif  param['type'] == 'ratio':
						type = 3
			n = Node_GPU(int(self.ptr)+shift,isLeaf,threshold,proba,tree.n_classes,
					idChildLeft, idChildRight, type,depth,
					c1,xm1,xM1,ym1,yM1,c2,xm2,xM2,ym2,yM2)

			assert(node_size == n.shift)
			shift += n.shift

	def free(self):
		self.ptr.free()

# ======================================================================
# Class Tree
# ======================================================================
class Tree:
	"""Class Tree """

	def __init__(self,featureFunction, splitter, min_samples_split, min_samples_leaf, max_depth, n_classes, max_leaf_nodes = -1):
		""" init """
		self.featureFunction = featureFunction
		self.splitter = splitter
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.max_depth = max_depth
		self.max_leaf_nodes = max_leaf_nodes
		self.n_classes = n_classes
		self.nodes=[]

	def release(self):
		self.splitter = None

	def _add_node(self, parent, is_left, is_leaf, feature,
							threshold, impurity, n_node_samples,
							stats, improvement, depth):
		""" _add_node """
		if(is_leaf):
			l = Leaf(parent, is_left, impurity, n_node_samples, stats, depth)
			self.nodes.append(l)
		else:
			s = Split(parent, is_left, feature, threshold, impurity,
						n_node_samples, improvement, stats, depth)
			self.nodes.append(s)

		node_id=len(self.nodes)-1
		
		if(parent > -1): # this node is not the tree root
			if(is_left):
				id=0
			else:
				id=1
			self.nodes[parent].childs[id] = node_id
		
		return node_id

	def depthFirstTreeBuilderCommon(self):
		"""Build a decision tree with depth First strategie """

		# Parameters
		splitter = self.splitter
		max_depth = self.max_depth
		min_samples_leaf = self.min_samples_leaf
		min_samples_split = self.min_samples_split

		n_node_samples = splitter.n_samples

		impurity = numpy.inf
		first = 1
		max_depth_seen = -1

		stack = []
		stackRecord=StackRecord(0, n_node_samples, 0, -1, 0, numpy.inf,[])
		# push root node onto stack
		stack.append(stackRecord)

		while stack: # stack is not empty
			stack_record = stack.pop()

			start = stack_record.start
			end = stack_record.end
			depth = stack_record.depth
			parent = stack_record.parent
			is_left = stack_record.is_left
			impurity = stack_record.impurity

			n_node_samples = end - start

			splitter.node_reset(start, end)

			is_leaf = ((depth >= max_depth) or
					   (n_node_samples < min_samples_split) or
					   (n_node_samples < 2 * min_samples_leaf))

			if first:
				impurity = splitter.node_impurity()
				first = 0

			is_leaf = is_leaf or (impurity <= 1e-7)

			split = SplitRecord()

			if not is_leaf:
				split = splitter.node_split(impurity)
				is_leaf = is_leaf or (split.pos >= end)

			node_id = self._add_node(parent, is_left, is_leaf, split.feature,
									 split.threshold, impurity, n_node_samples,
									 stack_record.stats, split.improvement, depth)

			if not is_leaf:
				# Push right child on stack
				stack.append(StackRecord(split.pos, end, depth + 1, node_id, 0,
								split.impurity_right, split.stats_right))

				# Push left child on stack
				stack.append(StackRecord(start, split.pos, depth + 1, node_id, 1,
								split.impurity_left, split.stats_left))

			if depth > max_depth_seen:
				max_depth_seen = depth

		self.max_depth = max_depth_seen

	def depthFirstTreeBuilder(self, X, y, sample_weight):
		"""Build a decision tree from the training set (X, y) with depth First strategie """
		# Recursive partition (without actual recursion)
		self.splitter.init(X, y, sample_weight)
		
		self.depthFirstTreeBuilderCommon()

	def depthFirstTreeBuilderImage(self, X, sample_index, y, sample_weight):
		"""Build a decision tree from the training set (X, y) with depth First strategie """

		# Recursive partition (without actual recursion)
		self.splitter.init_image(X, sample_index, y, sample_weight)
		
		self.depthFirstTreeBuilderCommon()

	def entangledTreeBuilderImage(self, X, sample_index, y, sample_weight, entangled):
		"""Build a decision tree from the training set (X, y) with entangled strategie """
		X = X.astype(numpy.float32)
		# Recursive partition (without actual recursion)
		self.splitter.init_image(X, sample_index, y, sample_weight)
		#TODO
		# Parameters
		splitter = self.splitter
		max_depth = self.max_depth
		min_samples_leaf = self.min_samples_leaf
		min_samples_split = self.min_samples_split

		n_node_samples = splitter.n_samples

		impurity = numpy.inf
		first = 1
		max_depth_seen = -1

		stack = []
		stack_next_level = []
		stackRecord=StackRecord(0, n_node_samples, 0, -1, 0, numpy.inf,[])
		# push root node onto stack
		stack_next_level.append(stackRecord)
		
		for index,depth_level in enumerate(entangled):
		
			stack = deepcopy(stack_next_level)
			stack_next_level = []
			
			while stack: # stack is not empty
				stack_record = stack.pop()

				start = stack_record.start
				end = stack_record.end
				depth = stack_record.depth
				parent = stack_record.parent
				is_left = stack_record.is_left
				impurity = stack_record.impurity

				n_node_samples = end - start

				splitter.node_reset(start, end)

				is_leaf = ((depth >= max_depth) or
							 (n_node_samples < min_samples_split) or
							 (n_node_samples < 2 * min_samples_leaf))

				if first:
					impurity = splitter.node_impurity()
					first = 0

				is_leaf = is_leaf or (impurity <= 1e-7)

				split = SplitRecord()

				if not is_leaf:
					split = splitter.node_split(impurity)
					is_leaf = is_leaf or (split.pos >= end)

				node_id = self._add_node(parent, is_left, is_leaf, split.feature,
										 split.threshold, impurity, n_node_samples,
										 stack_record.stats, split.improvement, depth)

				if not is_leaf:
					rc = StackRecord(split.pos, end, depth + 1, node_id, 0,
										split.impurity_right, split.stats_right)
					lc = StackRecord(start, split.pos, depth + 1, node_id, 1,
										split.impurity_left, split.stats_left)
					if depth < depth_level:
						stack.append(rc)
						stack.append(lc)
					else :
						stack_next_level.append(rc)
						stack_next_level.append(lc)

				if depth > max_depth_seen:
					max_depth_seen = depth
			
			if(len(stack_next_level)):
				print("####################### depth_level ", depth_level)
				#while stack
				#TODO compute proba
				print("compute proba 1")

				s_c, s_x, s_y = X.shape

				x_gpu = gpuarray.to_gpu(X)
				proba_gpu = gpuarray.zeros((self.n_classes,s_x,s_y),numpy.float32)
				w_x,w_y = self.featureFunction.width, self.featureFunction.height
				self.predict_image(x_gpu,proba_gpu, w_x, w_y, intermediate = True)

				print("compute proba 2")
				proba = proba_gpu.get()
				print(proba.shape)

				proba = proba.cumsum(2).cumsum(1)
				if index == 0:
					X = numpy.concatenate((X,proba))
				else :
					p = proba.shape[0]
					X = numpy.concatenate((X[:-p,:,:],proba))

			
				splitter.X = X
				self.featureFunction.nb_channels = X.shape[0]
				print(X.shape[0])

		self.max_depth = max_depth_seen
		print("####################### max_depth ", max_depth_seen)
		
	def bestFirstTreeBuilder(self, X, y):
		""" bestFirstTreeBuilder """
		#TODO
		print("bestFirstTreeBuilder is not yet implemented !")

	def bestFirstTreeBuilderImage(self, X_gpu, sample_index, y, sample_weight):
		"""Build a decision tree from the training set (X, y) with depth First strategie """
		#TODO
		print("bestFirstTreeBuilder is not yet implemented !")

	def predict(self, X):
		"""Predict target for X :
		Finds the terminal region (=leaf node) for each sample in X."""
		# Extract input
		n_samples = X.shape[0]
		n_bands = X.shape[1]

		# Initialize output
		out = numpy.zeros((n_samples,self.n_classes))
		Xtmp = numpy.ones((1,n_bands))
		for i in range(n_samples):
			node = self.nodes[0]
			Xtmp[0,:] = X[i]
			Xf = node.feature.evaluate(Xtmp)	
			# While node not a leaf
			while not node.isLeaf():
				if Xf[0] <= node.threshold:
					node = self.nodes[node.childs[0]]
				else:
					node = self.nodes[node.childs[1]]
				
			out[i,:] = node.stats
		return out

	def getGPUSourceModule(self, sizeX, sizeY):
		func_mod_template = Template("""

			// Macro for converting subscripts to linear index:
			// A = nb_channels
			// B,C = image X,Y
			#define INDEX3D(a, b, c) a*${sizeX}*${sizeY}+b*${sizeY}+c
			#define INDEX2D(a, b) a*${sizeY}+b

			struct Node {
				unsigned int isLeaf;
				float threshold;
				unsigned int idChildLeft, idChildRight;
				unsigned int depth;
				unsigned int type;
				int c1,xm1,xM1,ym1,yM1;
				int c2,xm2,xM2,ym2,yM2;
				float proba[${n_classes}];
			};

		__device__ float one(float *X, unsigned int idx, unsigned int idy , int c1,
								int dxm1, int dxM1,
								int dym1, int dyM1)
			{
				int xm1 = idx+dxm1;
				int xM1 = idx+dxM1;
				int ym1 = idy+dym1;
				int yM1 = idy+dyM1;
				
				xm1 = (xm1>=0)?xm1:0;
				ym1 = (ym1>=0)?ym1:0;
				
				xM1 = (xM1< ${sizeX})?xM1:(${sizeX}-1);
				yM1 = (yM1< ${sizeY})?yM1:(${sizeY}-1);
				
				float a = X[INDEX3D(c1,xm1,ym1)];
				float b = X[INDEX3D(c1,xM1,ym1)];
				float c = X[INDEX3D(c1,xm1,yM1)];
				float d = X[INDEX3D(c1,xM1,yM1)];
				return a - b - c + d;
			}

		__global__ void apply_tree(float *X, unsigned int *idTab, Node *nodeTab, unsigned int max_depth = 0)
			{
				// Obtain the linear index corresponding to the current thread:
				unsigned int idx = blockIdx.x*${n_threads_x}+threadIdx.x;
				unsigned int idy = blockIdx.y*${n_threads_y}+threadIdx.y;

				if(idx < ${sizeX} && idy < ${sizeY})
				{
					unsigned int node_id = idTab[INDEX2D(idx,idy)] ;
					Node n = nodeTab[node_id];
					while(!n.isLeaf or (max_depth > 0 and n.depth <= max_depth)) //we are in a split node
					{
						float r = 0;
						if(n.type == 0) // ONE
							r = one(X,idx, idy,n.c1,n.xm1,n.xM1,n.ym1,n.yM1);
						else if (n.type == 1) // SUM
							r = one(X,idx, idy,n.c1,n.xm1,n.xM1,n.ym1,n.yM1) + one(X,idx, idy,n.c2,n.xm2,n.xM2,n.ym2,n.yM2);
						else if (n.type == 2) // DIFF
							r = one(X,idx, idy,n.c1,n.xm1,n.xM1,n.ym1,n.yM1) - one(X,idx, idy,n.c2,n.xm2,n.xM2,n.ym2,n.yM2);
						else if (n.type == 3) // RATIO
						{
							r = one(X,idx, idy,n.c1,n.xm1,n.xM1,n.ym1,n.yM1) + one(X,idx, idy,n.c2,n.xm2,n.xM2,n.ym2,n.yM2);
							if(abs(r)>=0.00001)
								r = one(X,idx, idy,n.c1,n.xm1,n.xM1,n.ym1,n.yM1) - one(X,idx, idy,n.c2,n.xm2,n.xM2,n.ym2,n.yM2) / r;
							else
								r = 0;
						}
						if(r <= n.threshold)
							node_id = n.idChildLeft;
						else
							node_id = n.idChildRight;
						n = nodeTab[node_id];
					}
					idTab[INDEX2D(idx,idy)] = node_id;
				}
			}

		__global__ void add_proba(float *probaTab, unsigned int *idTab, Node *nodeTab)
			{
				// Obtain the linear index corresponding to the current thread:
				unsigned int idx = blockIdx.x*${n_threads_x}+threadIdx.x;
				unsigned int idy = blockIdx.y*${n_threads_y}+threadIdx.y;

				if(idx < ${sizeX} && idy < ${sizeY})
				{
					Node n = nodeTab[idTab[INDEX2D(idx,idy)]];
					for(unsigned int c =0; c< ${n_classes}; c++)
						probaTab[INDEX3D(c,idx,idy)] += n.proba[c];
				}
			}
		__global__ void concat_proba(float *X, unsigned int *idTab, Node *nodeTab, unsigned int shift)
			{
				// Obtain the linear index corresponding to the current thread:
				unsigned int idx = blockIdx.x*${n_threads_x}+threadIdx.x;
				unsigned int idy = blockIdx.y*${n_threads_y}+threadIdx.y;

				if(idx < ${sizeX} && idy < ${sizeY})
				{
					Node n = nodeTab[idTab[INDEX2D(idx,idy)]];
					for(unsigned int c = 0; c < ${n_classes}; c++)
						X[INDEX3D(c+shift,idx,idy)] = n.proba[c];
				}
			}
			""")

		

		max_threads_per_block = pycuda.autoinit.device.MAX_THREADS_PER_BLOCK
		max_threads = int(math.sqrt(max_threads_per_block))

		n_threads_x = min(max_threads,sizeX)
		n_threads_y = min(max_threads,sizeY)
		n_blocks_x = sizeX // n_threads_x +1;
		n_blocks_y = sizeY // n_threads_y +1;

		block_dim=(n_threads_x,n_threads_y,1)
		grid_dim=(n_blocks_x,n_blocks_y)

		func_mod = SourceModule(
					func_mod_template.substitute(
						n_threads_x=n_threads_x,
						n_threads_y=n_threads_y,
						sizeX=sizeX, sizeY=sizeY,
						n_classes = self.n_classes)
					)
		return func_mod,block_dim,grid_dim

	def predict_image(self, x_gpu, proba_gpu, w_x, w_y, intermediate = False, entangled=None):

		nbChannels, sizeX, sizeY = x_gpu.shape
		func_mod,block_dim,grid_dim = self.getGPUSourceModule(sizeX, sizeY)

		apply_tree = func_mod.get_function('apply_tree')
		add_proba = func_mod.get_function('add_proba')
		concat_proba = func_mod.get_function('concat_proba')

		#Copy data to GPU
		tree_gpu = Tree_GPU(self,intermediate=intermediate)
		
		id_gpu = gpuarray.zeros((sizeX,sizeY),numpy.uint32)
		
		if entangled is None:
			print("Normal")
			#Find the leaf of each samples
			apply_tree(x_gpu, id_gpu, tree_gpu.ptr, block=block_dim, grid=grid_dim)
			#Add Node's proba to global proba array
			add_proba(proba_gpu, id_gpu, tree_gpu.ptr, block=block_dim, grid=grid_dim)
		else:
			print("Entangled")
			X = x_gpu.get()
			for index,depth_level in enumerate(entangled):
				print("depth_level",depth_level)

				if(depth_level < self.max_depth):
				
					#Find the leaf of each samples until depth
					apply_tree(x_gpu, id_gpu, tree_gpu.ptr, numpy.uint32(depth_level), block=block_dim, grid=grid_dim)
					#Add intermediate proba at the end of X
					tmp = numpy.zeros((self.n_classes, sizeX, sizeY))
					tmp_gpu = gpuarray.to_gpu(tmp.astype(numpy.float32))
					add_proba(tmp_gpu, id_gpu, tree_gpu.ptr, block=block_dim, grid=grid_dim)
					tmp = tmp_gpu.get()
					p = tmp.shape[0]
					X = numpy.concatenate((X[:-p,:,:],tmp.cumsum(2).cumsum(1)))
					x_gpu = gpuarray.to_gpu(X)
					#TODO cumsum on proba !!
					#concat_proba(x_gpu, id_gpu, tree_gpu.ptr, numpy.uint32(nbChannels) , block=block_dim, grid=grid_dim)

			#Add Node's proba to global proba array
			add_proba(proba_gpu, id_gpu, tree_gpu.ptr, block=block_dim, grid=grid_dim)
					
					

	def getFeatureImportance(self, acc = None):
		if(acc == None):
			acc = self.featureFunction.getEmptyAccumulator()
		for node in self.nodes:
			if not node.isLeaf():
				#idLeft, idRight = node.childs
				#improvement = node.impurity - self.nodes[idLeft].impurity - self.nodes[idRight].impurity
				improvement = node.improvement
				node.feature.addFeatureImportance(improvement, acc)
		
		return acc

	def getFeatureImportanceByClass(self, acc = None):
		if(acc == None):
			acc = self.featureFunction.getEmptyAccumulatorByClass(self.n_classes)
		for node in self.nodes:
			if not node.isLeaf():
				#idLeft, idRight = node.childs
				#improvement = node.impurity - self.nodes[idLeft].impurity - self.nodes[idRight].impurity
				improvement = node.improvement
				if(len(node.stats) == self.n_classes):
					stats = numpy.asarray(node.stats)
					node.feature.addFeatureImportanceByClass(improvement, stats, acc)
		
		return acc

	def __repr__(self):
		out=""
		for node in self.nodes:
			out += node.__repr__() +'\r\n'
		return out
