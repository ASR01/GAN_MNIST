?? 
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-0-g3f878cff5b68??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
{
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?* 
shared_namedense_30/kernel
t
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
_output_shapes
:	d?*
dtype0
s
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_30/bias
l
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_21/gamma
?
0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_21/beta
?
/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_21/moving_mean
?
6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_21/moving_variance
?
:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes	
:?*
dtype0
|
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_31/kernel
u
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel* 
_output_shapes
:
??*
dtype0
s
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_31/bias
l
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_22/gamma
?
0batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_22/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_22/beta
?
/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_22/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_22/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_22/moving_mean
?
6batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_22/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_22/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_22/moving_variance
?
:batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_22/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_14/kernel
?
.conv2d_transpose_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_14/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_14/bias
?
,conv2d_transpose_14/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_14/bias*
_output_shapes
: *
dtype0
?
batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_23/gamma
?
0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_23/beta
?
/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_23/moving_mean
?
6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_23/moving_variance
?
:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_transpose_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_15/kernel
?
.conv2d_transpose_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_15/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_15/bias
?
,conv2d_transpose_15/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_15/bias*
_output_shapes
:*
dtype0
?
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
: *
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
: *
dtype0
?
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:@*
dtype0
|
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_28/kernel
u
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel* 
_output_shapes
:
??*
dtype0
s
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_28/bias
l
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes	
:?*
dtype0
{
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_29/kernel
t
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel*
_output_shapes
:	?*
dtype0
r
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_29/bias
k
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes
:*
dtype0
j
Adam/iter_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdam/iter_1
c
Adam/iter_1/Read/ReadVariableOpReadVariableOpAdam/iter_1*
_output_shapes
: *
dtype0	
n
Adam/beta_1_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1_1
g
!Adam/beta_1_1/Read/ReadVariableOpReadVariableOpAdam/beta_1_1*
_output_shapes
: *
dtype0
n
Adam/beta_2_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2_1
g
!Adam/beta_2_1/Read/ReadVariableOpReadVariableOpAdam/beta_2_1*
_output_shapes
: *
dtype0
l
Adam/decay_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/decay_1
e
 Adam/decay_1/Read/ReadVariableOpReadVariableOpAdam/decay_1*
_output_shapes
: *
dtype0
|
Adam/learning_rate_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/learning_rate_1
u
(Adam/learning_rate_1/Read/ReadVariableOpReadVariableOpAdam/learning_rate_1*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/dense_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*'
shared_nameAdam/dense_30/kernel/m
?
*Adam/dense_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/m*
_output_shapes
:	d?*
dtype0
?
Adam/dense_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_30/bias/m
z
(Adam/dense_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_21/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_21/gamma/m
?
7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_21/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_21/beta/m
?
6Adam/batch_normalization_21/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_31/kernel/m
?
*Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_31/bias/m
z
(Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_22/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_22/gamma/m
?
7Adam/batch_normalization_22/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_22/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_22/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_22/beta/m
?
6Adam/batch_normalization_22/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_22/beta/m*
_output_shapes	
:?*
dtype0
?
!Adam/conv2d_transpose_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_14/kernel/m
?
5Adam/conv2d_transpose_14/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_14/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_14/bias/m
?
3Adam/conv2d_transpose_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_14/bias/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_23/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_23/gamma/m
?
7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_23/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_23/beta/m
?
6Adam/batch_normalization_23/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/m*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_15/kernel/m
?
5Adam/conv2d_transpose_15/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_15/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_15/bias/m
?
3Adam/conv2d_transpose_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_15/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*'
shared_nameAdam/dense_30/kernel/v
?
*Adam/dense_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/v*
_output_shapes
:	d?*
dtype0
?
Adam/dense_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_30/bias/v
z
(Adam/dense_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_21/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_21/gamma/v
?
7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_21/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_21/beta/v
?
6Adam/batch_normalization_21/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_31/kernel/v
?
*Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_31/bias/v
z
(Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_22/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_22/gamma/v
?
7Adam/batch_normalization_22/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_22/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_22/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_22/beta/v
?
6Adam/batch_normalization_22/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_22/beta/v*
_output_shapes	
:?*
dtype0
?
!Adam/conv2d_transpose_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_14/kernel/v
?
5Adam/conv2d_transpose_14/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_14/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_transpose_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_14/bias/v
?
3Adam/conv2d_transpose_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_14/bias/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_23/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_23/gamma/v
?
7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_23/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_23/beta/v
?
6Adam/batch_normalization_23/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/v*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_15/kernel/v
?
5Adam/conv2d_transpose_15/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_15/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_15/bias/v
?
3Adam/conv2d_transpose_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_15/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_14/kernel/m
?
+Adam/conv2d_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_14/bias/m
{
)Adam/conv2d_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_15/kernel/m
?
+Adam/conv2d_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_15/bias/m
{
)Adam/conv2d_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_28/kernel/m
?
*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_28/bias/m
z
(Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_29/kernel/m
?
*Adam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/m
y
(Adam/dense_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_14/kernel/v
?
+Adam/conv2d_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_14/bias/v
{
)Adam/conv2d_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_15/kernel/v
?
+Adam/conv2d_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_15/bias/v
{
)Adam/conv2d_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_28/kernel/v
?
*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_28/bias/v
z
(Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_29/kernel/v
?
*Adam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/v
y
(Adam/dense_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
valueܽBؽ Bн
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures*
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer-8
layer_with_weights-5
layer-9
layer_with_weights-6
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
layer_with_weights-0
layer-0
layer-1
 layer_with_weights-1
 layer-2
!layer-3
"layer-4
#layer_with_weights-2
#layer-5
$layer-6
%layer_with_weights-3
%layer-7
&	optimizer
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses*
?
-iter

.beta_1

/beta_2
	0decay
1learning_rate2m?3m?4m?5m?8m?9m?:m?;m?>m??m?@m?Am?Dm?Em?2v?3v?4v?5v?8v?9v?:v?;v?>v??v?@v?Av?Dv?Ev?*
?
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
B16
C17
D18
E19
F20
G21
H22
I23
J24
K25
L26
M27*
j
20
31
42
53
84
95
:6
;7
>8
?9
@10
A11
D12
E13*
* 
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 

Sserving_default* 
?

2kernel
3bias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses*
?
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses* 
?
`axis
	4gamma
5beta
6moving_mean
7moving_variance
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses*
?

8kernel
9bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses*
?
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses* 
?
saxis
	:gamma
;beta
<moving_mean
=moving_variance
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses*
?
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
?

>kernel
?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
	?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

Dkernel
Ebias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
B16
C17
D18
E19*
j
20
31
42
53
84
95
:6
;7
>8
?9
@10
A11
D12
E13*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
?

Fkernel
Gbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

Hkernel
Ibias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

Jkernel
Kbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

Lkernel
Mbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateFm?Gm?Hm?Im?Jm?Km?Lm?Mm?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?*
<
F0
G1
H2
I3
J4
K5
L6
M7*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_30/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_30/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_21/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_21/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"batch_normalization_21/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_normalization_21/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_31/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_31/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_22/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_22/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"batch_normalization_22/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&batch_normalization_22/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_transpose_14/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_14/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEbatch_normalization_23/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_23/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"batch_normalization_23/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&batch_normalization_23/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_transpose_15/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_15/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv2d_14/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_14/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv2d_15/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_15/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_28/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_28/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_29/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_29/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
j
60
71
<2
=3
B4
C5
F6
G7
H8
I9
J10
K11
L12
M13*

0
1*

?0*
* 
* 
* 

20
31*

20
31*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses* 
* 
* 
* 
 
40
51
62
73*

40
51*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 

80
91*

80
91*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 
* 
* 
* 
 
:0
;1
<2
=3*

:0
;1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 

>0
?1*

>0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
 
@0
A1
B2
C3*

@0
A1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

D0
E1*

D0
E1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
.
60
71
<2
=3
B4
C5*
Z
0
1
2
3
4
5
6
7
8
9
10
11*
* 
* 
* 

F0
G1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

H0
I1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

J0
K1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

L0
M1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEAdam/iter_1>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/beta_1_1@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/beta_2_1@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/decay_1?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/learning_rate_1Glayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
<
F0
G1
H2
I3
J4
K5
L6
M7*
<
0
1
 2
!3
"4
#5
$6
%7*

?0
?1*
* 
* 
<

?total

?count
?	variables
?	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

60
71*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

<0
=1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

B0
C1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

F0
G1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

H0
I1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

J0
K1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

L0
M1*
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
jd
VARIABLE_VALUEtotal_1Ilayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcount_1Ilayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
jd
VARIABLE_VALUEtotal_2Ilayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcount_2Ilayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
rl
VARIABLE_VALUEAdam/dense_30/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_30/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/batch_normalization_21/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_31/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_31/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/batch_normalization_22/gamma/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/batch_normalization_22/beta/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/conv2d_transpose_14/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d_transpose_14/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE#Adam/batch_normalization_23/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_23/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/conv2d_transpose_15/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d_transpose_15/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_30/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_30/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/batch_normalization_21/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_31/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_31/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/batch_normalization_22/gamma/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/batch_normalization_22/beta/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/conv2d_transpose_14/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d_transpose_14/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE#Adam/batch_normalization_23/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/batch_normalization_23/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/conv2d_transpose_15/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d_transpose_15/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_14/kernel/mXvariables/20/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_14/bias/mXvariables/21/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_15/kernel/mXvariables/22/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_15/bias/mXvariables/23/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_28/kernel/mXvariables/24/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_28/bias/mXvariables/25/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_29/kernel/mXvariables/26/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_29/bias/mXvariables/27/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_14/kernel/vXvariables/20/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_14/bias/vXvariables/21/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_15/kernel/vXvariables/22/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_15/bias/vXvariables/23/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_28/kernel/vXvariables/24/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_28/bias/vXvariables/25/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_29/kernel/vXvariables/26/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_29/bias/vXvariables/27/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
#serving_default_sequential_22_inputPlaceholder*'
_output_shapes
:?????????d*
dtype0*
shape:?????????d
?
StatefulPartitionedCallStatefulPartitionedCall#serving_default_sequential_22_inputdense_30/kerneldense_30/bias&batch_normalization_21/moving_variancebatch_normalization_21/gamma"batch_normalization_21/moving_meanbatch_normalization_21/betadense_31/kerneldense_31/bias&batch_normalization_22/moving_variancebatch_normalization_22/gamma"batch_normalization_22/moving_meanbatch_normalization_22/betaconv2d_transpose_14/kernelconv2d_transpose_14/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_varianceconv2d_transpose_15/kernelconv2d_transpose_15/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_1085393
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?!
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp0batch_normalization_22/gamma/Read/ReadVariableOp/batch_normalization_22/beta/Read/ReadVariableOp6batch_normalization_22/moving_mean/Read/ReadVariableOp:batch_normalization_22/moving_variance/Read/ReadVariableOp.conv2d_transpose_14/kernel/Read/ReadVariableOp,conv2d_transpose_14/bias/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp.conv2d_transpose_15/kernel/Read/ReadVariableOp,conv2d_transpose_15/bias/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOpAdam/iter_1/Read/ReadVariableOp!Adam/beta_1_1/Read/ReadVariableOp!Adam/beta_2_1/Read/ReadVariableOp Adam/decay_1/Read/ReadVariableOp(Adam/learning_rate_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*Adam/dense_30/kernel/m/Read/ReadVariableOp(Adam/dense_30/bias/m/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_21/beta/m/Read/ReadVariableOp*Adam/dense_31/kernel/m/Read/ReadVariableOp(Adam/dense_31/bias/m/Read/ReadVariableOp7Adam/batch_normalization_22/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_22/beta/m/Read/ReadVariableOp5Adam/conv2d_transpose_14/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_14/bias/m/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_23/beta/m/Read/ReadVariableOp5Adam/conv2d_transpose_15/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_15/bias/m/Read/ReadVariableOp*Adam/dense_30/kernel/v/Read/ReadVariableOp(Adam/dense_30/bias/v/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_21/beta/v/Read/ReadVariableOp*Adam/dense_31/kernel/v/Read/ReadVariableOp(Adam/dense_31/bias/v/Read/ReadVariableOp7Adam/batch_normalization_22/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_22/beta/v/Read/ReadVariableOp5Adam/conv2d_transpose_14/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_14/bias/v/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_23/beta/v/Read/ReadVariableOp5Adam/conv2d_transpose_15/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_15/bias/v/Read/ReadVariableOp+Adam/conv2d_14/kernel/m/Read/ReadVariableOp)Adam/conv2d_14/bias/m/Read/ReadVariableOp+Adam/conv2d_15/kernel/m/Read/ReadVariableOp)Adam/conv2d_15/bias/m/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp(Adam/dense_28/bias/m/Read/ReadVariableOp*Adam/dense_29/kernel/m/Read/ReadVariableOp(Adam/dense_29/bias/m/Read/ReadVariableOp+Adam/conv2d_14/kernel/v/Read/ReadVariableOp)Adam/conv2d_14/bias/v/Read/ReadVariableOp+Adam/conv2d_15/kernel/v/Read/ReadVariableOp)Adam/conv2d_15/bias/v/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOp(Adam/dense_28/bias/v/Read/ReadVariableOp*Adam/dense_29/kernel/v/Read/ReadVariableOp(Adam/dense_29/bias/v/Read/ReadVariableOpConst*e
Tin^
\2Z		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_1086663
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_30/kerneldense_30/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_variancedense_31/kerneldense_31/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_varianceconv2d_transpose_14/kernelconv2d_transpose_14/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_varianceconv2d_transpose_15/kernelconv2d_transpose_15/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/biasAdam/iter_1Adam/beta_1_1Adam/beta_2_1Adam/decay_1Adam/learning_rate_1totalcounttotal_1count_1total_2count_2Adam/dense_30/kernel/mAdam/dense_30/bias/m#Adam/batch_normalization_21/gamma/m"Adam/batch_normalization_21/beta/mAdam/dense_31/kernel/mAdam/dense_31/bias/m#Adam/batch_normalization_22/gamma/m"Adam/batch_normalization_22/beta/m!Adam/conv2d_transpose_14/kernel/mAdam/conv2d_transpose_14/bias/m#Adam/batch_normalization_23/gamma/m"Adam/batch_normalization_23/beta/m!Adam/conv2d_transpose_15/kernel/mAdam/conv2d_transpose_15/bias/mAdam/dense_30/kernel/vAdam/dense_30/bias/v#Adam/batch_normalization_21/gamma/v"Adam/batch_normalization_21/beta/vAdam/dense_31/kernel/vAdam/dense_31/bias/v#Adam/batch_normalization_22/gamma/v"Adam/batch_normalization_22/beta/v!Adam/conv2d_transpose_14/kernel/vAdam/conv2d_transpose_14/bias/v#Adam/batch_normalization_23/gamma/v"Adam/batch_normalization_23/beta/v!Adam/conv2d_transpose_15/kernel/vAdam/conv2d_transpose_15/bias/vAdam/conv2d_14/kernel/mAdam/conv2d_14/bias/mAdam/conv2d_15/kernel/mAdam/conv2d_15/bias/mAdam/dense_28/kernel/mAdam/dense_28/bias/mAdam/dense_29/kernel/mAdam/dense_29/bias/mAdam/conv2d_14/kernel/vAdam/conv2d_14/bias/vAdam/conv2d_15/kernel/vAdam/conv2d_15/bias/vAdam/dense_28/kernel/vAdam/dense_28/bias/vAdam/dense_29/kernel/vAdam/dense_29/bias/v*d
Tin]
[2Y*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_1086937??
?

?
/__inference_sequential_21_layer_call_fn_1084181
conv2d_14_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084162o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_14_input
?
?
/__inference_sequential_22_layer_call_fn_1085483

inputs
unknown:	d?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?$

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*0
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_22_layer_call_and_return_conditional_losses_1083860w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
E__inference_dense_31_layer_call_and_return_conditional_losses_1085983

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?#
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084364
conv2d_14_input+
conv2d_14_1084339: 
conv2d_14_1084341: +
conv2d_15_1084345: @
conv2d_15_1084347:@$
dense_28_1084352:
??
dense_28_1084354:	?#
dense_29_1084358:	?
dense_29_1084360:
identity??!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallconv2d_14_inputconv2d_14_1084339conv2d_14_1084341*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_14_layer_call_and_return_conditional_losses_1084077?
leaky_re_lu_21/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1084088?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0conv2d_15_1084345conv2d_15_1084347*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1084100?
leaky_re_lu_22/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1084111?
flatten_7/PartitionedCallPartitionedCall'leaky_re_lu_22/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_1084119?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_28_1084352dense_28_1084354*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_1084131?
leaky_re_lu_23/PartitionedCallPartitionedCall)dense_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1084142?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0dense_29_1084358dense_29_1084360*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_1084155x
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_14_input
?
K
/__inference_activation_30_layer_call_fn_1086139

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_30_layer_call_and_return_conditional_losses_1083640h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1086356

inputs
identityH
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:??????????`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_activation_31_layer_call_and_return_conditional_losses_1086258

inputs
identityN
TanhTanhinputs*
T0*/
_output_shapes
:?????????X
IdentityIdentityTanh:y:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1083298

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
J__inference_sequential_22_layer_call_and_return_conditional_losses_1085739

inputs:
'dense_30_matmul_readvariableop_resource:	d?7
(dense_30_biasadd_readvariableop_resource:	?M
>batch_normalization_21_assignmovingavg_readvariableop_resource:	?O
@batch_normalization_21_assignmovingavg_1_readvariableop_resource:	?K
<batch_normalization_21_batchnorm_mul_readvariableop_resource:	?G
8batch_normalization_21_batchnorm_readvariableop_resource:	?;
'dense_31_matmul_readvariableop_resource:
??7
(dense_31_biasadd_readvariableop_resource:	?M
>batch_normalization_22_assignmovingavg_readvariableop_resource:	?O
@batch_normalization_22_assignmovingavg_1_readvariableop_resource:	?K
<batch_normalization_22_batchnorm_mul_readvariableop_resource:	?G
8batch_normalization_22_batchnorm_readvariableop_resource:	?V
<conv2d_transpose_14_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_14_biasadd_readvariableop_resource: <
.batch_normalization_23_readvariableop_resource: >
0batch_normalization_23_readvariableop_1_resource: M
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_15_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_15_biasadd_readvariableop_resource:
identity??&batch_normalization_21/AssignMovingAvg?5batch_normalization_21/AssignMovingAvg/ReadVariableOp?(batch_normalization_21/AssignMovingAvg_1?7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_21/batchnorm/ReadVariableOp?3batch_normalization_21/batchnorm/mul/ReadVariableOp?&batch_normalization_22/AssignMovingAvg?5batch_normalization_22/AssignMovingAvg/ReadVariableOp?(batch_normalization_22/AssignMovingAvg_1?7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_22/batchnorm/ReadVariableOp?3batch_normalization_22/batchnorm/mul/ReadVariableOp?%batch_normalization_23/AssignNewValue?'batch_normalization_23/AssignNewValue_1?6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_23/ReadVariableOp?'batch_normalization_23/ReadVariableOp_1?*conv2d_transpose_14/BiasAdd/ReadVariableOp?3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?*conv2d_transpose_15/BiasAdd/ReadVariableOp?3conv2d_transpose_15/conv2d_transpose/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0|
dense_30/MatMulMatMulinputs&dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????h
activation_28/ReluReludense_30/BiasAdd:output:0*
T0*(
_output_shapes
:??????????
5batch_normalization_21/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
#batch_normalization_21/moments/meanMean activation_28/Relu:activations:0>batch_normalization_21/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
+batch_normalization_21/moments/StopGradientStopGradient,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes
:	??
0batch_normalization_21/moments/SquaredDifferenceSquaredDifference activation_28/Relu:activations:04batch_normalization_21/moments/StopGradient:output:0*
T0*(
_output_shapes
:???????????
9batch_normalization_21/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
'batch_normalization_21/moments/varianceMean4batch_normalization_21/moments/SquaredDifference:z:0Bbatch_normalization_21/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
&batch_normalization_21/moments/SqueezeSqueeze,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 ?
(batch_normalization_21/moments/Squeeze_1Squeeze0batch_normalization_21/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 q
,batch_normalization_21/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
5batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*batch_normalization_21/AssignMovingAvg/subSub=batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_21/moments/Squeeze:output:0*
T0*
_output_shapes	
:??
*batch_normalization_21/AssignMovingAvg/mulMul.batch_normalization_21/AssignMovingAvg/sub:z:05batch_normalization_21/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
&batch_normalization_21/AssignMovingAvgAssignSubVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource.batch_normalization_21/AssignMovingAvg/mul:z:06^batch_normalization_21/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_21/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,batch_normalization_21/AssignMovingAvg_1/subSub?batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_21/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:??
,batch_normalization_21/AssignMovingAvg_1/mulMul0batch_normalization_21/AssignMovingAvg_1/sub:z:07batch_normalization_21/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
(batch_normalization_21/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource0batch_normalization_21/AssignMovingAvg_1/mul:z:08^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
$batch_normalization_21/batchnorm/addAddV21batch_normalization_21/moments/Squeeze_1:output:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes	
:??
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
&batch_normalization_21/batchnorm/mul_1Mul activation_28/Relu:activations:0(batch_normalization_21/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
&batch_normalization_21/batchnorm/mul_2Mul/batch_normalization_21/moments/Squeeze:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$batch_normalization_21/batchnorm/subSub7batch_normalization_21/batchnorm/ReadVariableOp:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_31/MatMulMatMul*batch_normalization_21/batchnorm/add_1:z:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????h
activation_29/ReluReludense_31/BiasAdd:output:0*
T0*(
_output_shapes
:??????????
5batch_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
#batch_normalization_22/moments/meanMean activation_29/Relu:activations:0>batch_normalization_22/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
+batch_normalization_22/moments/StopGradientStopGradient,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes
:	??
0batch_normalization_22/moments/SquaredDifferenceSquaredDifference activation_29/Relu:activations:04batch_normalization_22/moments/StopGradient:output:0*
T0*(
_output_shapes
:???????????
9batch_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
'batch_normalization_22/moments/varianceMean4batch_normalization_22/moments/SquaredDifference:z:0Bbatch_normalization_22/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
&batch_normalization_22/moments/SqueezeSqueeze,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 ?
(batch_normalization_22/moments/Squeeze_1Squeeze0batch_normalization_22/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 q
,batch_normalization_22/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
5batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
*batch_normalization_22/AssignMovingAvg/subSub=batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_22/moments/Squeeze:output:0*
T0*
_output_shapes	
:??
*batch_normalization_22/AssignMovingAvg/mulMul.batch_normalization_22/AssignMovingAvg/sub:z:05batch_normalization_22/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
&batch_normalization_22/AssignMovingAvgAssignSubVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource.batch_normalization_22/AssignMovingAvg/mul:z:06^batch_normalization_22/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_22/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,batch_normalization_22/AssignMovingAvg_1/subSub?batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_22/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:??
,batch_normalization_22/AssignMovingAvg_1/mulMul0batch_normalization_22/AssignMovingAvg_1/sub:z:07batch_normalization_22/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
(batch_normalization_22/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource0batch_normalization_22/AssignMovingAvg_1/mul:z:08^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
$batch_normalization_22/batchnorm/addAddV21batch_normalization_22/moments/Squeeze_1:output:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:??
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
&batch_normalization_22/batchnorm/mul_1Mul activation_29/Relu:activations:0(batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
&batch_normalization_22/batchnorm/mul_2Mul/batch_normalization_22/moments/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$batch_normalization_22/batchnorm/subSub7batch_normalization_22/batchnorm/ReadVariableOp:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????i
reshape_7/ShapeShape*batch_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
reshape_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_7/strided_sliceStridedSlicereshape_7/Shape:output:0&reshape_7/strided_slice/stack:output:0(reshape_7/strided_slice/stack_1:output:0(reshape_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_7/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
reshape_7/Reshape/shapePack reshape_7/strided_slice:output:0"reshape_7/Reshape/shape/1:output:0"reshape_7/Reshape/shape/2:output:0"reshape_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_7/ReshapeReshape*batch_normalization_22/batchnorm/add_1:z:0 reshape_7/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@c
conv2d_transpose_14/ShapeShapereshape_7/Reshape:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_14/strided_sliceStridedSlice"conv2d_transpose_14/Shape:output:00conv2d_transpose_14/strided_slice/stack:output:02conv2d_transpose_14/strided_slice/stack_1:output:02conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_14/stackPack*conv2d_transpose_14/strided_slice:output:0$conv2d_transpose_14/stack/1:output:0$conv2d_transpose_14/stack/2:output:0$conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_14/strided_slice_1StridedSlice"conv2d_transpose_14/stack:output:02conv2d_transpose_14/strided_slice_1/stack:output:04conv2d_transpose_14/strided_slice_1/stack_1:output:04conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
$conv2d_transpose_14/conv2d_transposeConv2DBackpropInput"conv2d_transpose_14/stack:output:0;conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0reshape_7/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
*conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_14/BiasAddBiasAdd-conv2d_transpose_14/conv2d_transpose:output:02conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? z
activation_30/ReluRelu$conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3 activation_30/Relu:activations:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_23/AssignNewValueAssignVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource4batch_normalization_23/FusedBatchNormV3:batch_mean:07^batch_normalization_23/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_23/AssignNewValue_1AssignVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_23/FusedBatchNormV3:batch_variance:09^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0t
conv2d_transpose_15/ShapeShape+batch_normalization_23/FusedBatchNormV3:y:0*
T0*
_output_shapes
:q
'conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_15/strided_sliceStridedSlice"conv2d_transpose_15/Shape:output:00conv2d_transpose_15/strided_slice/stack:output:02conv2d_transpose_15/strided_slice/stack_1:output:02conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_15/stackPack*conv2d_transpose_15/strided_slice:output:0$conv2d_transpose_15/stack/1:output:0$conv2d_transpose_15/stack/2:output:0$conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_15/strided_slice_1StridedSlice"conv2d_transpose_15/stack:output:02conv2d_transpose_15/strided_slice_1/stack:output:04conv2d_transpose_15/strided_slice_1/stack_1:output:04conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_15/conv2d_transposeConv2DBackpropInput"conv2d_transpose_15/stack:output:0;conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
*conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_15/BiasAddBiasAdd-conv2d_transpose_15/conv2d_transpose:output:02conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
activation_31/TanhTanh$conv2d_transpose_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????m
IdentityIdentityactivation_31/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????

NoOpNoOp'^batch_normalization_21/AssignMovingAvg6^batch_normalization_21/AssignMovingAvg/ReadVariableOp)^batch_normalization_21/AssignMovingAvg_18^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_21/batchnorm/ReadVariableOp4^batch_normalization_21/batchnorm/mul/ReadVariableOp'^batch_normalization_22/AssignMovingAvg6^batch_normalization_22/AssignMovingAvg/ReadVariableOp)^batch_normalization_22/AssignMovingAvg_18^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_22/batchnorm/ReadVariableOp4^batch_normalization_22/batchnorm/mul/ReadVariableOp&^batch_normalization_23/AssignNewValue(^batch_normalization_23/AssignNewValue_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1+^conv2d_transpose_14/BiasAdd/ReadVariableOp4^conv2d_transpose_14/conv2d_transpose/ReadVariableOp+^conv2d_transpose_15/BiasAdd/ReadVariableOp4^conv2d_transpose_15/conv2d_transpose/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_21/AssignMovingAvg&batch_normalization_21/AssignMovingAvg2n
5batch_normalization_21/AssignMovingAvg/ReadVariableOp5batch_normalization_21/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_21/AssignMovingAvg_1(batch_normalization_21/AssignMovingAvg_12r
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2P
&batch_normalization_22/AssignMovingAvg&batch_normalization_22/AssignMovingAvg2n
5batch_normalization_22/AssignMovingAvg/ReadVariableOp5batch_normalization_22/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_22/AssignMovingAvg_1(batch_normalization_22/AssignMovingAvg_12r
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2N
%batch_normalization_23/AssignNewValue%batch_normalization_23/AssignNewValue2R
'batch_normalization_23/AssignNewValue_1'batch_normalization_23/AssignNewValue_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12X
*conv2d_transpose_14/BiasAdd/ReadVariableOp*conv2d_transpose_14/BiasAdd/ReadVariableOp2j
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp3conv2d_transpose_14/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_15/BiasAdd/ReadVariableOp*conv2d_transpose_15/BiasAdd/ReadVariableOp2j
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp3conv2d_transpose_15/conv2d_transpose/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1084142

inputs
identityH
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:??????????`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_activation_29_layer_call_and_return_conditional_losses_1083603

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:??????????[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?#
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084296

inputs+
conv2d_14_1084271: 
conv2d_14_1084273: +
conv2d_15_1084277: @
conv2d_15_1084279:@$
dense_28_1084284:
??
dense_28_1084286:	?#
dense_29_1084290:	?
dense_29_1084292:
identity??!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_14_1084271conv2d_14_1084273*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_14_layer_call_and_return_conditional_losses_1084077?
leaky_re_lu_21/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1084088?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0conv2d_15_1084277conv2d_15_1084279*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1084100?
leaky_re_lu_22/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1084111?
flatten_7/PartitionedCallPartitionedCall'leaky_re_lu_22/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_1084119?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_28_1084284dense_28_1084286*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_1084131?
leaky_re_lu_23/PartitionedCallPartitionedCall)dense_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1084142?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0dense_29_1084290dense_29_1084292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_1084155x
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
P__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_1083428

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?
J__inference_sequential_23_layer_call_and_return_conditional_losses_1085158

inputsH
5sequential_22_dense_30_matmul_readvariableop_resource:	d?E
6sequential_22_dense_30_biasadd_readvariableop_resource:	?U
Fsequential_22_batch_normalization_21_batchnorm_readvariableop_resource:	?Y
Jsequential_22_batch_normalization_21_batchnorm_mul_readvariableop_resource:	?W
Hsequential_22_batch_normalization_21_batchnorm_readvariableop_1_resource:	?W
Hsequential_22_batch_normalization_21_batchnorm_readvariableop_2_resource:	?I
5sequential_22_dense_31_matmul_readvariableop_resource:
??E
6sequential_22_dense_31_biasadd_readvariableop_resource:	?U
Fsequential_22_batch_normalization_22_batchnorm_readvariableop_resource:	?Y
Jsequential_22_batch_normalization_22_batchnorm_mul_readvariableop_resource:	?W
Hsequential_22_batch_normalization_22_batchnorm_readvariableop_1_resource:	?W
Hsequential_22_batch_normalization_22_batchnorm_readvariableop_2_resource:	?d
Jsequential_22_conv2d_transpose_14_conv2d_transpose_readvariableop_resource: @O
Asequential_22_conv2d_transpose_14_biasadd_readvariableop_resource: J
<sequential_22_batch_normalization_23_readvariableop_resource: L
>sequential_22_batch_normalization_23_readvariableop_1_resource: [
Msequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_resource: ]
Osequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource: d
Jsequential_22_conv2d_transpose_15_conv2d_transpose_readvariableop_resource: O
Asequential_22_conv2d_transpose_15_biasadd_readvariableop_resource:P
6sequential_21_conv2d_14_conv2d_readvariableop_resource: E
7sequential_21_conv2d_14_biasadd_readvariableop_resource: P
6sequential_21_conv2d_15_conv2d_readvariableop_resource: @E
7sequential_21_conv2d_15_biasadd_readvariableop_resource:@I
5sequential_21_dense_28_matmul_readvariableop_resource:
??E
6sequential_21_dense_28_biasadd_readvariableop_resource:	?H
5sequential_21_dense_29_matmul_readvariableop_resource:	?D
6sequential_21_dense_29_biasadd_readvariableop_resource:
identity??.sequential_21/conv2d_14/BiasAdd/ReadVariableOp?-sequential_21/conv2d_14/Conv2D/ReadVariableOp?.sequential_21/conv2d_15/BiasAdd/ReadVariableOp?-sequential_21/conv2d_15/Conv2D/ReadVariableOp?-sequential_21/dense_28/BiasAdd/ReadVariableOp?,sequential_21/dense_28/MatMul/ReadVariableOp?-sequential_21/dense_29/BiasAdd/ReadVariableOp?,sequential_21/dense_29/MatMul/ReadVariableOp?=sequential_22/batch_normalization_21/batchnorm/ReadVariableOp??sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_1??sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_2?Asequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOp?=sequential_22/batch_normalization_22/batchnorm/ReadVariableOp??sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_1??sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_2?Asequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOp?Dsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?Fsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?3sequential_22/batch_normalization_23/ReadVariableOp?5sequential_22/batch_normalization_23/ReadVariableOp_1?8sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOp?Asequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?8sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOp?Asequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?-sequential_22/dense_30/BiasAdd/ReadVariableOp?,sequential_22/dense_30/MatMul/ReadVariableOp?-sequential_22/dense_31/BiasAdd/ReadVariableOp?,sequential_22/dense_31/MatMul/ReadVariableOp?
,sequential_22/dense_30/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_30_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
sequential_22/dense_30/MatMulMatMulinputs4sequential_22/dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_22/dense_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_22/dense_30/BiasAddBiasAdd'sequential_22/dense_30/MatMul:product:05sequential_22/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
 sequential_22/activation_28/ReluRelu'sequential_22/dense_30/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
=sequential_22/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOpFsequential_22_batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0y
4sequential_22/batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
2sequential_22/batch_normalization_21/batchnorm/addAddV2Esequential_22/batch_normalization_21/batchnorm/ReadVariableOp:value:0=sequential_22/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_21/batchnorm/RsqrtRsqrt6sequential_22/batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes	
:??
Asequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_22_batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2sequential_22/batch_normalization_21/batchnorm/mulMul8sequential_22/batch_normalization_21/batchnorm/Rsqrt:y:0Isequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_21/batchnorm/mul_1Mul.sequential_22/activation_28/Relu:activations:06sequential_22/batch_normalization_21/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
?sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_22_batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
4sequential_22/batch_normalization_21/batchnorm/mul_2MulGsequential_22/batch_normalization_21/batchnorm/ReadVariableOp_1:value:06sequential_22/batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
?sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_22_batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
2sequential_22/batch_normalization_21/batchnorm/subSubGsequential_22/batch_normalization_21/batchnorm/ReadVariableOp_2:value:08sequential_22/batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_21/batchnorm/add_1AddV28sequential_22/batch_normalization_21/batchnorm/mul_1:z:06sequential_22/batch_normalization_21/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
,sequential_22/dense_31/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_22/dense_31/MatMulMatMul8sequential_22/batch_normalization_21/batchnorm/add_1:z:04sequential_22/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_22/dense_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_22/dense_31/BiasAddBiasAdd'sequential_22/dense_31/MatMul:product:05sequential_22/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
 sequential_22/activation_29/ReluRelu'sequential_22/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
=sequential_22/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOpFsequential_22_batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0y
4sequential_22/batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
2sequential_22/batch_normalization_22/batchnorm/addAddV2Esequential_22/batch_normalization_22/batchnorm/ReadVariableOp:value:0=sequential_22/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_22/batchnorm/RsqrtRsqrt6sequential_22/batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:??
Asequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_22_batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2sequential_22/batch_normalization_22/batchnorm/mulMul8sequential_22/batch_normalization_22/batchnorm/Rsqrt:y:0Isequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_22/batchnorm/mul_1Mul.sequential_22/activation_29/Relu:activations:06sequential_22/batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
?sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_22_batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
4sequential_22/batch_normalization_22/batchnorm/mul_2MulGsequential_22/batch_normalization_22/batchnorm/ReadVariableOp_1:value:06sequential_22/batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
?sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_22_batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
2sequential_22/batch_normalization_22/batchnorm/subSubGsequential_22/batch_normalization_22/batchnorm/ReadVariableOp_2:value:08sequential_22/batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_22/batchnorm/add_1AddV28sequential_22/batch_normalization_22/batchnorm/mul_1:z:06sequential_22/batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
sequential_22/reshape_7/ShapeShape8sequential_22/batch_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:u
+sequential_22/reshape_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_22/reshape_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_22/reshape_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%sequential_22/reshape_7/strided_sliceStridedSlice&sequential_22/reshape_7/Shape:output:04sequential_22/reshape_7/strided_slice/stack:output:06sequential_22/reshape_7/strided_slice/stack_1:output:06sequential_22/reshape_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'sequential_22/reshape_7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :i
'sequential_22/reshape_7/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :i
'sequential_22/reshape_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
%sequential_22/reshape_7/Reshape/shapePack.sequential_22/reshape_7/strided_slice:output:00sequential_22/reshape_7/Reshape/shape/1:output:00sequential_22/reshape_7/Reshape/shape/2:output:00sequential_22/reshape_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
sequential_22/reshape_7/ReshapeReshape8sequential_22/batch_normalization_22/batchnorm/add_1:z:0.sequential_22/reshape_7/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@
'sequential_22/conv2d_transpose_14/ShapeShape(sequential_22/reshape_7/Reshape:output:0*
T0*
_output_shapes
:
5sequential_22/conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7sequential_22/conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential_22/conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_22/conv2d_transpose_14/strided_sliceStridedSlice0sequential_22/conv2d_transpose_14/Shape:output:0>sequential_22/conv2d_transpose_14/strided_slice/stack:output:0@sequential_22/conv2d_transpose_14/strided_slice/stack_1:output:0@sequential_22/conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_22/conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_22/conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_22/conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_22/conv2d_transpose_14/stackPack8sequential_22/conv2d_transpose_14/strided_slice:output:02sequential_22/conv2d_transpose_14/stack/1:output:02sequential_22/conv2d_transpose_14/stack/2:output:02sequential_22/conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:?
7sequential_22/conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9sequential_22/conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential_22/conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1sequential_22/conv2d_transpose_14/strided_slice_1StridedSlice0sequential_22/conv2d_transpose_14/stack:output:0@sequential_22/conv2d_transpose_14/strided_slice_1/stack:output:0Bsequential_22/conv2d_transpose_14/strided_slice_1/stack_1:output:0Bsequential_22/conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Asequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_22_conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
2sequential_22/conv2d_transpose_14/conv2d_transposeConv2DBackpropInput0sequential_22/conv2d_transpose_14/stack:output:0Isequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0(sequential_22/reshape_7/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
8sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOpAsequential_22_conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
)sequential_22/conv2d_transpose_14/BiasAddBiasAdd;sequential_22/conv2d_transpose_14/conv2d_transpose:output:0@sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
 sequential_22/activation_30/ReluRelu2sequential_22/conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
3sequential_22/batch_normalization_23/ReadVariableOpReadVariableOp<sequential_22_batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype0?
5sequential_22/batch_normalization_23/ReadVariableOp_1ReadVariableOp>sequential_22_batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Dsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Fsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5sequential_22/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3.sequential_22/activation_30/Relu:activations:0;sequential_22/batch_normalization_23/ReadVariableOp:value:0=sequential_22/batch_normalization_23/ReadVariableOp_1:value:0Lsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
'sequential_22/conv2d_transpose_15/ShapeShape9sequential_22/batch_normalization_23/FusedBatchNormV3:y:0*
T0*
_output_shapes
:
5sequential_22/conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7sequential_22/conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential_22/conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_22/conv2d_transpose_15/strided_sliceStridedSlice0sequential_22/conv2d_transpose_15/Shape:output:0>sequential_22/conv2d_transpose_15/strided_slice/stack:output:0@sequential_22/conv2d_transpose_15/strided_slice/stack_1:output:0@sequential_22/conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_22/conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_22/conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_22/conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
'sequential_22/conv2d_transpose_15/stackPack8sequential_22/conv2d_transpose_15/strided_slice:output:02sequential_22/conv2d_transpose_15/stack/1:output:02sequential_22/conv2d_transpose_15/stack/2:output:02sequential_22/conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:?
7sequential_22/conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9sequential_22/conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential_22/conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1sequential_22/conv2d_transpose_15/strided_slice_1StridedSlice0sequential_22/conv2d_transpose_15/stack:output:0@sequential_22/conv2d_transpose_15/strided_slice_1/stack:output:0Bsequential_22/conv2d_transpose_15/strided_slice_1/stack_1:output:0Bsequential_22/conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Asequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_22_conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
2sequential_22/conv2d_transpose_15/conv2d_transposeConv2DBackpropInput0sequential_22/conv2d_transpose_15/stack:output:0Isequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:09sequential_22/batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
8sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOpAsequential_22_conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)sequential_22/conv2d_transpose_15/BiasAddBiasAdd;sequential_22/conv2d_transpose_15/conv2d_transpose:output:0@sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
 sequential_22/activation_31/TanhTanh2sequential_22/conv2d_transpose_15/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
-sequential_21/conv2d_14/Conv2D/ReadVariableOpReadVariableOp6sequential_21_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential_21/conv2d_14/Conv2DConv2D$sequential_22/activation_31/Tanh:y:05sequential_21/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
.sequential_21/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp7sequential_21_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_21/conv2d_14/BiasAddBiasAdd'sequential_21/conv2d_14/Conv2D:output:06sequential_21/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
&sequential_21/leaky_re_lu_21/LeakyRelu	LeakyRelu(sequential_21/conv2d_14/BiasAdd:output:0*/
_output_shapes
:????????? ?
-sequential_21/conv2d_15/Conv2D/ReadVariableOpReadVariableOp6sequential_21_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
sequential_21/conv2d_15/Conv2DConv2D4sequential_21/leaky_re_lu_21/LeakyRelu:activations:05sequential_21/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
.sequential_21/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp7sequential_21_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_21/conv2d_15/BiasAddBiasAdd'sequential_21/conv2d_15/Conv2D:output:06sequential_21/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
&sequential_21/leaky_re_lu_22/LeakyRelu	LeakyRelu(sequential_21/conv2d_15/BiasAdd:output:0*/
_output_shapes
:?????????@n
sequential_21/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
sequential_21/flatten_7/ReshapeReshape4sequential_21/leaky_re_lu_22/LeakyRelu:activations:0&sequential_21/flatten_7/Const:output:0*
T0*(
_output_shapes
:???????????
,sequential_21/dense_28/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_21/dense_28/MatMulMatMul(sequential_21/flatten_7/Reshape:output:04sequential_21/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_21/dense_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_21/dense_28/BiasAddBiasAdd'sequential_21/dense_28/MatMul:product:05sequential_21/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&sequential_21/leaky_re_lu_23/LeakyRelu	LeakyRelu'sequential_21/dense_28/BiasAdd:output:0*(
_output_shapes
:???????????
,sequential_21/dense_29/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential_21/dense_29/MatMulMatMul4sequential_21/leaky_re_lu_23/LeakyRelu:activations:04sequential_21/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-sequential_21/dense_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_21/dense_29/BiasAddBiasAdd'sequential_21/dense_29/MatMul:product:05sequential_21/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_21/dense_29/SigmoidSigmoid'sequential_21/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:?????????q
IdentityIdentity"sequential_21/dense_29/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^sequential_21/conv2d_14/BiasAdd/ReadVariableOp.^sequential_21/conv2d_14/Conv2D/ReadVariableOp/^sequential_21/conv2d_15/BiasAdd/ReadVariableOp.^sequential_21/conv2d_15/Conv2D/ReadVariableOp.^sequential_21/dense_28/BiasAdd/ReadVariableOp-^sequential_21/dense_28/MatMul/ReadVariableOp.^sequential_21/dense_29/BiasAdd/ReadVariableOp-^sequential_21/dense_29/MatMul/ReadVariableOp>^sequential_22/batch_normalization_21/batchnorm/ReadVariableOp@^sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_1@^sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_2B^sequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOp>^sequential_22/batch_normalization_22/batchnorm/ReadVariableOp@^sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_1@^sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_2B^sequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOpE^sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOpG^sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_14^sequential_22/batch_normalization_23/ReadVariableOp6^sequential_22/batch_normalization_23/ReadVariableOp_19^sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOpB^sequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOp9^sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOpB^sequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOp.^sequential_22/dense_30/BiasAdd/ReadVariableOp-^sequential_22/dense_30/MatMul/ReadVariableOp.^sequential_22/dense_31/BiasAdd/ReadVariableOp-^sequential_22/dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.sequential_21/conv2d_14/BiasAdd/ReadVariableOp.sequential_21/conv2d_14/BiasAdd/ReadVariableOp2^
-sequential_21/conv2d_14/Conv2D/ReadVariableOp-sequential_21/conv2d_14/Conv2D/ReadVariableOp2`
.sequential_21/conv2d_15/BiasAdd/ReadVariableOp.sequential_21/conv2d_15/BiasAdd/ReadVariableOp2^
-sequential_21/conv2d_15/Conv2D/ReadVariableOp-sequential_21/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_21/dense_28/BiasAdd/ReadVariableOp-sequential_21/dense_28/BiasAdd/ReadVariableOp2\
,sequential_21/dense_28/MatMul/ReadVariableOp,sequential_21/dense_28/MatMul/ReadVariableOp2^
-sequential_21/dense_29/BiasAdd/ReadVariableOp-sequential_21/dense_29/BiasAdd/ReadVariableOp2\
,sequential_21/dense_29/MatMul/ReadVariableOp,sequential_21/dense_29/MatMul/ReadVariableOp2~
=sequential_22/batch_normalization_21/batchnorm/ReadVariableOp=sequential_22/batch_normalization_21/batchnorm/ReadVariableOp2?
?sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_1?sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_12?
?sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_2?sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_22?
Asequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOpAsequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOp2~
=sequential_22/batch_normalization_22/batchnorm/ReadVariableOp=sequential_22/batch_normalization_22/batchnorm/ReadVariableOp2?
?sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_1?sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_12?
?sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_2?sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_22?
Asequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOpAsequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOp2?
Dsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOpDsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2?
Fsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Fsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12j
3sequential_22/batch_normalization_23/ReadVariableOp3sequential_22/batch_normalization_23/ReadVariableOp2n
5sequential_22/batch_normalization_23/ReadVariableOp_15sequential_22/batch_normalization_23/ReadVariableOp_12t
8sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOp8sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOp2?
Asequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOpAsequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOp2t
8sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOp8sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOp2?
Asequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOpAsequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOp2^
-sequential_22/dense_30/BiasAdd/ReadVariableOp-sequential_22/dense_30/BiasAdd/ReadVariableOp2\
,sequential_22/dense_30/MatMul/ReadVariableOp,sequential_22/dense_30/MatMul/ReadVariableOp2^
-sequential_22/dense_31/BiasAdd/ReadVariableOp-sequential_22/dense_31/BiasAdd/ReadVariableOp2\
,sequential_22/dense_31/MatMul/ReadVariableOp,sequential_22/dense_31/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_21_layer_call_fn_1085910

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1083298p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1084111

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
/__inference_sequential_23_layer_call_fn_1085014

inputs
unknown:	d?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?$

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:$

unknown_19: 

unknown_20: $

unknown_21: @

unknown_22:@

unknown_23:
??

unknown_24:	?

unknown_25:	?

unknown_26:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_23_layer_call_and_return_conditional_losses_1084642o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
/__inference_sequential_21_layer_call_fn_1085787

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084296o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_28_layer_call_and_return_conditional_losses_1084131

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1085964

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_23_layer_call_fn_1086170

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1083488?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
J__inference_sequential_22_layer_call_and_return_conditional_losses_1085597

inputs:
'dense_30_matmul_readvariableop_resource:	d?7
(dense_30_biasadd_readvariableop_resource:	?G
8batch_normalization_21_batchnorm_readvariableop_resource:	?K
<batch_normalization_21_batchnorm_mul_readvariableop_resource:	?I
:batch_normalization_21_batchnorm_readvariableop_1_resource:	?I
:batch_normalization_21_batchnorm_readvariableop_2_resource:	?;
'dense_31_matmul_readvariableop_resource:
??7
(dense_31_biasadd_readvariableop_resource:	?G
8batch_normalization_22_batchnorm_readvariableop_resource:	?K
<batch_normalization_22_batchnorm_mul_readvariableop_resource:	?I
:batch_normalization_22_batchnorm_readvariableop_1_resource:	?I
:batch_normalization_22_batchnorm_readvariableop_2_resource:	?V
<conv2d_transpose_14_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_14_biasadd_readvariableop_resource: <
.batch_normalization_23_readvariableop_resource: >
0batch_normalization_23_readvariableop_1_resource: M
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_15_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_15_biasadd_readvariableop_resource:
identity??/batch_normalization_21/batchnorm/ReadVariableOp?1batch_normalization_21/batchnorm/ReadVariableOp_1?1batch_normalization_21/batchnorm/ReadVariableOp_2?3batch_normalization_21/batchnorm/mul/ReadVariableOp?/batch_normalization_22/batchnorm/ReadVariableOp?1batch_normalization_22/batchnorm/ReadVariableOp_1?1batch_normalization_22/batchnorm/ReadVariableOp_2?3batch_normalization_22/batchnorm/mul/ReadVariableOp?6batch_normalization_23/FusedBatchNormV3/ReadVariableOp?8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_23/ReadVariableOp?'batch_normalization_23/ReadVariableOp_1?*conv2d_transpose_14/BiasAdd/ReadVariableOp?3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?*conv2d_transpose_15/BiasAdd/ReadVariableOp?3conv2d_transpose_15/conv2d_transpose/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0|
dense_30/MatMulMatMulinputs&dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????h
activation_28/ReluReludense_30/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0k
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
$batch_normalization_21/batchnorm/addAddV27batch_normalization_21/batchnorm/ReadVariableOp:value:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes	
:??
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
&batch_normalization_21/batchnorm/mul_1Mul activation_28/Relu:activations:0(batch_normalization_21/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
1batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_21/batchnorm/mul_2Mul9batch_normalization_21/batchnorm/ReadVariableOp_1:value:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
1batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
$batch_normalization_21/batchnorm/subSub9batch_normalization_21/batchnorm/ReadVariableOp_2:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_31/MatMulMatMul*batch_normalization_21/batchnorm/add_1:z:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????h
activation_29/ReluReludense_31/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
$batch_normalization_22/batchnorm/addAddV27batch_normalization_22/batchnorm/ReadVariableOp:value:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:??
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
&batch_normalization_22/batchnorm/mul_1Mul activation_29/Relu:activations:0(batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
1batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
&batch_normalization_22/batchnorm/mul_2Mul9batch_normalization_22/batchnorm/ReadVariableOp_1:value:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
1batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
$batch_normalization_22/batchnorm/subSub9batch_normalization_22/batchnorm/ReadVariableOp_2:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????i
reshape_7/ShapeShape*batch_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
reshape_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_7/strided_sliceStridedSlicereshape_7/Shape:output:0&reshape_7/strided_slice/stack:output:0(reshape_7/strided_slice/stack_1:output:0(reshape_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_7/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
reshape_7/Reshape/shapePack reshape_7/strided_slice:output:0"reshape_7/Reshape/shape/1:output:0"reshape_7/Reshape/shape/2:output:0"reshape_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_7/ReshapeReshape*batch_normalization_22/batchnorm/add_1:z:0 reshape_7/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@c
conv2d_transpose_14/ShapeShapereshape_7/Reshape:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_14/strided_sliceStridedSlice"conv2d_transpose_14/Shape:output:00conv2d_transpose_14/strided_slice/stack:output:02conv2d_transpose_14/strided_slice/stack_1:output:02conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_14/stackPack*conv2d_transpose_14/strided_slice:output:0$conv2d_transpose_14/stack/1:output:0$conv2d_transpose_14/stack/2:output:0$conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_14/strided_slice_1StridedSlice"conv2d_transpose_14/stack:output:02conv2d_transpose_14/strided_slice_1/stack:output:04conv2d_transpose_14/strided_slice_1/stack_1:output:04conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
$conv2d_transpose_14/conv2d_transposeConv2DBackpropInput"conv2d_transpose_14/stack:output:0;conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0reshape_7/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
*conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_14/BiasAddBiasAdd-conv2d_transpose_14/conv2d_transpose:output:02conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? z
activation_30/ReluRelu$conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3 activation_30/Relu:activations:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( t
conv2d_transpose_15/ShapeShape+batch_normalization_23/FusedBatchNormV3:y:0*
T0*
_output_shapes
:q
'conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_15/strided_sliceStridedSlice"conv2d_transpose_15/Shape:output:00conv2d_transpose_15/strided_slice/stack:output:02conv2d_transpose_15/strided_slice/stack_1:output:02conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_15/stackPack*conv2d_transpose_15/strided_slice:output:0$conv2d_transpose_15/stack/1:output:0$conv2d_transpose_15/stack/2:output:0$conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_15/strided_slice_1StridedSlice"conv2d_transpose_15/stack:output:02conv2d_transpose_15/strided_slice_1/stack:output:04conv2d_transpose_15/strided_slice_1/stack_1:output:04conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_15/conv2d_transposeConv2DBackpropInput"conv2d_transpose_15/stack:output:0;conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
*conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_15/BiasAddBiasAdd-conv2d_transpose_15/conv2d_transpose:output:02conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
activation_31/TanhTanh$conv2d_transpose_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????m
IdentityIdentityactivation_31/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^batch_normalization_21/batchnorm/ReadVariableOp2^batch_normalization_21/batchnorm/ReadVariableOp_12^batch_normalization_21/batchnorm/ReadVariableOp_24^batch_normalization_21/batchnorm/mul/ReadVariableOp0^batch_normalization_22/batchnorm/ReadVariableOp2^batch_normalization_22/batchnorm/ReadVariableOp_12^batch_normalization_22/batchnorm/ReadVariableOp_24^batch_normalization_22/batchnorm/mul/ReadVariableOp7^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1+^conv2d_transpose_14/BiasAdd/ReadVariableOp4^conv2d_transpose_14/conv2d_transpose/ReadVariableOp+^conv2d_transpose_15/BiasAdd/ReadVariableOp4^conv2d_transpose_15/conv2d_transpose/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2f
1batch_normalization_21/batchnorm/ReadVariableOp_11batch_normalization_21/batchnorm/ReadVariableOp_12f
1batch_normalization_21/batchnorm/ReadVariableOp_21batch_normalization_21/batchnorm/ReadVariableOp_22j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2f
1batch_normalization_22/batchnorm/ReadVariableOp_11batch_normalization_22/batchnorm/ReadVariableOp_12f
1batch_normalization_22/batchnorm/ReadVariableOp_21batch_normalization_22/batchnorm/ReadVariableOp_22j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12X
*conv2d_transpose_14/BiasAdd/ReadVariableOp*conv2d_transpose_14/BiasAdd/ReadVariableOp2j
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp3conv2d_transpose_14/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_15/BiasAdd/ReadVariableOp*conv2d_transpose_15/BiasAdd/ReadVariableOp2j
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp3conv2d_transpose_15/conv2d_transpose/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?=
?

J__inference_sequential_22_layer_call_and_return_conditional_losses_1083860

inputs#
dense_30_1083807:	d?
dense_30_1083809:	?-
batch_normalization_21_1083813:	?-
batch_normalization_21_1083815:	?-
batch_normalization_21_1083817:	?-
batch_normalization_21_1083819:	?$
dense_31_1083822:
??
dense_31_1083824:	?-
batch_normalization_22_1083828:	?-
batch_normalization_22_1083830:	?-
batch_normalization_22_1083832:	?-
batch_normalization_22_1083834:	?5
conv2d_transpose_14_1083838: @)
conv2d_transpose_14_1083840: ,
batch_normalization_23_1083844: ,
batch_normalization_23_1083846: ,
batch_normalization_23_1083848: ,
batch_normalization_23_1083850: 5
conv2d_transpose_15_1083853: )
conv2d_transpose_15_1083855:
identity??.batch_normalization_21/StatefulPartitionedCall?.batch_normalization_22/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCallinputsdense_30_1083807dense_30_1083809*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_30_layer_call_and_return_conditional_losses_1083560?
activation_28/PartitionedCallPartitionedCall)dense_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_28_layer_call_and_return_conditional_losses_1083571?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall&activation_28/PartitionedCall:output:0batch_normalization_21_1083813batch_normalization_21_1083815batch_normalization_21_1083817batch_normalization_21_1083819*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1083298?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0dense_31_1083822dense_31_1083824*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_1083592?
activation_29/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_29_layer_call_and_return_conditional_losses_1083603?
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall&activation_29/PartitionedCall:output:0batch_normalization_22_1083828batch_normalization_22_1083830batch_normalization_22_1083832batch_normalization_22_1083834*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1083380?
reshape_7/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_reshape_7_layer_call_and_return_conditional_losses_1083628?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv2d_transpose_14_1083838conv2d_transpose_14_1083840*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_1083428?
activation_30/PartitionedCallPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_30_layer_call_and_return_conditional_losses_1083640?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall&activation_30/PartitionedCall:output:0batch_normalization_23_1083844batch_normalization_23_1083846batch_normalization_23_1083848batch_normalization_23_1083850*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1083488?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0conv2d_transpose_15_1083853conv2d_transpose_15_1083855*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_1083536?
activation_31/PartitionedCallPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_31_layer_call_and_return_conditional_losses_1083661}
IdentityIdentity&activation_31/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?(
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1085855

inputsB
(conv2d_14_conv2d_readvariableop_resource: 7
)conv2d_14_biasadd_readvariableop_resource: B
(conv2d_15_conv2d_readvariableop_resource: @7
)conv2d_15_biasadd_readvariableop_resource:@;
'dense_28_matmul_readvariableop_resource:
??7
(dense_28_biasadd_readvariableop_resource:	?:
'dense_29_matmul_readvariableop_resource:	?6
(dense_29_biasadd_readvariableop_resource:
identity?? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?dense_28/MatMul/ReadVariableOp?dense_29/BiasAdd/ReadVariableOp?dense_29/MatMul/ReadVariableOp?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_14/Conv2DConv2Dinputs'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? r
leaky_re_lu_21/LeakyRelu	LeakyReluconv2d_14/BiasAdd:output:0*/
_output_shapes
:????????? ?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_15/Conv2DConv2D&leaky_re_lu_21/LeakyRelu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@r
leaky_re_lu_22/LeakyRelu	LeakyReluconv2d_15/BiasAdd:output:0*/
_output_shapes
:?????????@`
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
flatten_7/ReshapeReshape&leaky_re_lu_22/LeakyRelu:activations:0flatten_7/Const:output:0*
T0*(
_output_shapes
:???????????
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_28/MatMulMatMulflatten_7/Reshape:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????j
leaky_re_lu_23/LeakyRelu	LeakyReludense_28/BiasAdd:output:0*(
_output_shapes
:???????????
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_29/MatMulMatMul&leaky_re_lu_23/LeakyRelu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_29/SigmoidSigmoiddense_29/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_29/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_reshape_7_layer_call_and_return_conditional_losses_1086092

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_30_layer_call_and_return_conditional_losses_1085874

inputs1
matmul_readvariableop_resource:	d?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?#
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084162

inputs+
conv2d_14_1084078: 
conv2d_14_1084080: +
conv2d_15_1084101: @
conv2d_15_1084103:@$
dense_28_1084132:
??
dense_28_1084134:	?#
dense_29_1084156:	?
dense_29_1084158:
identity??!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_14_1084078conv2d_14_1084080*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_14_layer_call_and_return_conditional_losses_1084077?
leaky_re_lu_21/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1084088?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0conv2d_15_1084101conv2d_15_1084103*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1084100?
leaky_re_lu_22/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1084111?
flatten_7/PartitionedCallPartitionedCall'leaky_re_lu_22/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_1084119?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_28_1084132dense_28_1084134*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_1084131?
leaky_re_lu_23/PartitionedCallPartitionedCall)dense_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1084142?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0dense_29_1084156dense_29_1084158*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_1084155x
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_15_layer_call_fn_1086215

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_1083536?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
f
J__inference_activation_29_layer_call_and_return_conditional_losses_1085993

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:??????????[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_14_layer_call_fn_1086267

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_14_layer_call_and_return_conditional_losses_1084077w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_21_layer_call_fn_1085897

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1083251p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_23_layer_call_fn_1084762
sequential_22_input
unknown:	d?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?$

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:$

unknown_19: 

unknown_20: $

unknown_21: @

unknown_22:@

unknown_23:
??

unknown_24:	?

unknown_25:	?

unknown_26:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsequential_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_23_layer_call_and_return_conditional_losses_1084642o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
'
_output_shapes
:?????????d
-
_user_specified_namesequential_22_input
?%
?
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1083380

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
K
/__inference_activation_28_layer_call_fn_1085879

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_28_layer_call_and_return_conditional_losses_1083571a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_22_layer_call_fn_1083948
dense_30_input
unknown:	d?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?$

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_30_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*0
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_22_layer_call_and_return_conditional_losses_1083860w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????d
(
_user_specified_namedense_30_input
?
?
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1083488

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
K
/__inference_activation_29_layer_call_fn_1085988

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_29_layer_call_and_return_conditional_losses_1083603a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
/__inference_sequential_21_layer_call_fn_1084336
conv2d_14_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084296o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_14_input
? 
?
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_1083536

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_23_layer_call_fn_1086351

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1084142a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_31_layer_call_and_return_conditional_losses_1083592

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_14_layer_call_and_return_conditional_losses_1086277

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_23_layer_call_fn_1086157

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1083457?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_1085393
sequential_22_input
unknown:	d?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?$

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:$

unknown_19: 

unknown_20: $

unknown_21: @

unknown_22:@

unknown_23:
??

unknown_24:	?

unknown_25:	?

unknown_26:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsequential_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_1083227o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
'
_output_shapes
:?????????d
-
_user_specified_namesequential_22_input
?	
?
E__inference_dense_28_layer_call_and_return_conditional_losses_1086346

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1086039

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_14_layer_call_fn_1086101

inputs!
unknown: @
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_1083428?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_dense_31_layer_call_fn_1085973

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_1083592p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1086316

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1086287

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
G
+__inference_flatten_7_layer_call_fn_1086321

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_1084119a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
f
J__inference_activation_30_layer_call_and_return_conditional_losses_1086144

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_22_layer_call_fn_1086019

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1083380p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
K
/__inference_activation_31_layer_call_fn_1086253

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_31_layer_call_and_return_conditional_losses_1083661h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1084088

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
/__inference_sequential_23_layer_call_fn_1084517
sequential_22_input
unknown:	d?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?$

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:$

unknown_19: 

unknown_20: $

unknown_21: @

unknown_22:@

unknown_23:
??

unknown_24:	?

unknown_25:	?

unknown_26:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsequential_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_23_layer_call_and_return_conditional_losses_1084458o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
'
_output_shapes
:?????????d
-
_user_specified_namesequential_22_input
? 
?
P__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_1086134

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
/__inference_sequential_23_layer_call_fn_1084953

inputs
unknown:	d?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?$

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:$

unknown_19: 

unknown_20: $

unknown_21: @

unknown_22:@

unknown_23:
??

unknown_24:	?

unknown_25:	?

unknown_26:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_23_layer_call_and_return_conditional_losses_1084458o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1083251

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1085930

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_15_layer_call_fn_1086296

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1084100w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1086188

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
b
F__inference_flatten_7_layer_call_and_return_conditional_losses_1084119

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?

J__inference_sequential_23_layer_call_and_return_conditional_losses_1084886
sequential_22_input(
sequential_22_1084827:	d?$
sequential_22_1084829:	?$
sequential_22_1084831:	?$
sequential_22_1084833:	?$
sequential_22_1084835:	?$
sequential_22_1084837:	?)
sequential_22_1084839:
??$
sequential_22_1084841:	?$
sequential_22_1084843:	?$
sequential_22_1084845:	?$
sequential_22_1084847:	?$
sequential_22_1084849:	?/
sequential_22_1084851: @#
sequential_22_1084853: #
sequential_22_1084855: #
sequential_22_1084857: #
sequential_22_1084859: #
sequential_22_1084861: /
sequential_22_1084863: #
sequential_22_1084865:/
sequential_21_1084868: #
sequential_21_1084870: /
sequential_21_1084872: @#
sequential_21_1084874:@)
sequential_21_1084876:
??$
sequential_21_1084878:	?(
sequential_21_1084880:	?#
sequential_21_1084882:
identity??%sequential_21/StatefulPartitionedCall?%sequential_22/StatefulPartitionedCall?
%sequential_22/StatefulPartitionedCallStatefulPartitionedCallsequential_22_inputsequential_22_1084827sequential_22_1084829sequential_22_1084831sequential_22_1084833sequential_22_1084835sequential_22_1084837sequential_22_1084839sequential_22_1084841sequential_22_1084843sequential_22_1084845sequential_22_1084847sequential_22_1084849sequential_22_1084851sequential_22_1084853sequential_22_1084855sequential_22_1084857sequential_22_1084859sequential_22_1084861sequential_22_1084863sequential_22_1084865* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*0
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_22_layer_call_and_return_conditional_losses_1083860?
%sequential_21/StatefulPartitionedCallStatefulPartitionedCall.sequential_22/StatefulPartitionedCall:output:0sequential_21_1084868sequential_21_1084870sequential_21_1084872sequential_21_1084874sequential_21_1084876sequential_21_1084878sequential_21_1084880sequential_21_1084882*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084296}
IdentityIdentity.sequential_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^sequential_21/StatefulPartitionedCall&^sequential_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%sequential_21/StatefulPartitionedCall%sequential_21/StatefulPartitionedCall2N
%sequential_22/StatefulPartitionedCall%sequential_22/StatefulPartitionedCall:\ X
'
_output_shapes
:?????????d
-
_user_specified_namesequential_22_input
?%
?
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1086073

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_activation_31_layer_call_and_return_conditional_losses_1083661

inputs
identityN
TanhTanhinputs*
T0*/
_output_shapes
:?????????X
IdentityIdentityTanh:y:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_flatten_7_layer_call_and_return_conditional_losses_1086327

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?

J__inference_sequential_23_layer_call_and_return_conditional_losses_1084824
sequential_22_input(
sequential_22_1084765:	d?$
sequential_22_1084767:	?$
sequential_22_1084769:	?$
sequential_22_1084771:	?$
sequential_22_1084773:	?$
sequential_22_1084775:	?)
sequential_22_1084777:
??$
sequential_22_1084779:	?$
sequential_22_1084781:	?$
sequential_22_1084783:	?$
sequential_22_1084785:	?$
sequential_22_1084787:	?/
sequential_22_1084789: @#
sequential_22_1084791: #
sequential_22_1084793: #
sequential_22_1084795: #
sequential_22_1084797: #
sequential_22_1084799: /
sequential_22_1084801: #
sequential_22_1084803:/
sequential_21_1084806: #
sequential_21_1084808: /
sequential_21_1084810: @#
sequential_21_1084812:@)
sequential_21_1084814:
??$
sequential_21_1084816:	?(
sequential_21_1084818:	?#
sequential_21_1084820:
identity??%sequential_21/StatefulPartitionedCall?%sequential_22/StatefulPartitionedCall?
%sequential_22/StatefulPartitionedCallStatefulPartitionedCallsequential_22_inputsequential_22_1084765sequential_22_1084767sequential_22_1084769sequential_22_1084771sequential_22_1084773sequential_22_1084775sequential_22_1084777sequential_22_1084779sequential_22_1084781sequential_22_1084783sequential_22_1084785sequential_22_1084787sequential_22_1084789sequential_22_1084791sequential_22_1084793sequential_22_1084795sequential_22_1084797sequential_22_1084799sequential_22_1084801sequential_22_1084803* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_22_layer_call_and_return_conditional_losses_1083664?
%sequential_21/StatefulPartitionedCallStatefulPartitionedCall.sequential_22/StatefulPartitionedCall:output:0sequential_21_1084806sequential_21_1084808sequential_21_1084810sequential_21_1084812sequential_21_1084814sequential_21_1084816sequential_21_1084818sequential_21_1084820*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084162}
IdentityIdentity.sequential_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^sequential_21/StatefulPartitionedCall&^sequential_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%sequential_21/StatefulPartitionedCall%sequential_21/StatefulPartitionedCall2N
%sequential_22/StatefulPartitionedCall%sequential_22/StatefulPartitionedCall:\ X
'
_output_shapes
:?????????d
-
_user_specified_namesequential_22_input
?
L
0__inference_leaky_re_lu_21_layer_call_fn_1086282

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1084088h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?'
 __inference__traced_save_1086663
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop;
7savev2_batch_normalization_21_gamma_read_readvariableop:
6savev2_batch_normalization_21_beta_read_readvariableopA
=savev2_batch_normalization_21_moving_mean_read_readvariableopE
Asavev2_batch_normalization_21_moving_variance_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop;
7savev2_batch_normalization_22_gamma_read_readvariableop:
6savev2_batch_normalization_22_beta_read_readvariableopA
=savev2_batch_normalization_22_moving_mean_read_readvariableopE
Asavev2_batch_normalization_22_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_14_kernel_read_readvariableop7
3savev2_conv2d_transpose_14_bias_read_readvariableop;
7savev2_batch_normalization_23_gamma_read_readvariableop:
6savev2_batch_normalization_23_beta_read_readvariableopA
=savev2_batch_normalization_23_moving_mean_read_readvariableopE
Asavev2_batch_normalization_23_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_15_kernel_read_readvariableop7
3savev2_conv2d_transpose_15_bias_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableop*
&savev2_adam_iter_1_read_readvariableop	,
(savev2_adam_beta_1_1_read_readvariableop,
(savev2_adam_beta_2_1_read_readvariableop+
'savev2_adam_decay_1_read_readvariableop3
/savev2_adam_learning_rate_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop5
1savev2_adam_dense_30_kernel_m_read_readvariableop3
/savev2_adam_dense_30_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_21_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_21_beta_m_read_readvariableop5
1savev2_adam_dense_31_kernel_m_read_readvariableop3
/savev2_adam_dense_31_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_22_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_22_beta_m_read_readvariableop@
<savev2_adam_conv2d_transpose_14_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_14_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_23_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_23_beta_m_read_readvariableop@
<savev2_adam_conv2d_transpose_15_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_15_bias_m_read_readvariableop5
1savev2_adam_dense_30_kernel_v_read_readvariableop3
/savev2_adam_dense_30_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_21_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_21_beta_v_read_readvariableop5
1savev2_adam_dense_31_kernel_v_read_readvariableop3
/savev2_adam_dense_31_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_22_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_22_beta_v_read_readvariableop@
<savev2_adam_conv2d_transpose_14_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_14_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_23_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_23_beta_v_read_readvariableop@
<savev2_adam_conv2d_transpose_15_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_15_bias_v_read_readvariableop6
2savev2_adam_conv2d_14_kernel_m_read_readvariableop4
0savev2_adam_conv2d_14_bias_m_read_readvariableop6
2savev2_adam_conv2d_15_kernel_m_read_readvariableop4
0savev2_adam_conv2d_15_bias_m_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop3
/savev2_adam_dense_28_bias_m_read_readvariableop5
1savev2_adam_dense_29_kernel_m_read_readvariableop3
/savev2_adam_dense_29_bias_m_read_readvariableop6
2savev2_adam_conv2d_14_kernel_v_read_readvariableop4
0savev2_adam_conv2d_14_bias_v_read_readvariableop6
2savev2_adam_conv2d_15_kernel_v_read_readvariableop4
0savev2_adam_conv2d_15_bias_v_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop3
/savev2_adam_dense_28_bias_v_read_readvariableop5
1savev2_adam_dense_29_kernel_v_read_readvariableop3
/savev2_adam_dense_29_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?+
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*?+
value?+B?*YB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/20/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/21/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/22/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/23/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/24/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/25/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/26/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/27/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/20/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/21/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/22/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/23/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/24/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/25/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/26/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/27/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*?
value?B?YB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?%
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop7savev2_batch_normalization_21_gamma_read_readvariableop6savev2_batch_normalization_21_beta_read_readvariableop=savev2_batch_normalization_21_moving_mean_read_readvariableopAsavev2_batch_normalization_21_moving_variance_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop7savev2_batch_normalization_22_gamma_read_readvariableop6savev2_batch_normalization_22_beta_read_readvariableop=savev2_batch_normalization_22_moving_mean_read_readvariableopAsavev2_batch_normalization_22_moving_variance_read_readvariableop5savev2_conv2d_transpose_14_kernel_read_readvariableop3savev2_conv2d_transpose_14_bias_read_readvariableop7savev2_batch_normalization_23_gamma_read_readvariableop6savev2_batch_normalization_23_beta_read_readvariableop=savev2_batch_normalization_23_moving_mean_read_readvariableopAsavev2_batch_normalization_23_moving_variance_read_readvariableop5savev2_conv2d_transpose_15_kernel_read_readvariableop3savev2_conv2d_transpose_15_bias_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop&savev2_adam_iter_1_read_readvariableop(savev2_adam_beta_1_1_read_readvariableop(savev2_adam_beta_2_1_read_readvariableop'savev2_adam_decay_1_read_readvariableop/savev2_adam_learning_rate_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop1savev2_adam_dense_30_kernel_m_read_readvariableop/savev2_adam_dense_30_bias_m_read_readvariableop>savev2_adam_batch_normalization_21_gamma_m_read_readvariableop=savev2_adam_batch_normalization_21_beta_m_read_readvariableop1savev2_adam_dense_31_kernel_m_read_readvariableop/savev2_adam_dense_31_bias_m_read_readvariableop>savev2_adam_batch_normalization_22_gamma_m_read_readvariableop=savev2_adam_batch_normalization_22_beta_m_read_readvariableop<savev2_adam_conv2d_transpose_14_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_14_bias_m_read_readvariableop>savev2_adam_batch_normalization_23_gamma_m_read_readvariableop=savev2_adam_batch_normalization_23_beta_m_read_readvariableop<savev2_adam_conv2d_transpose_15_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_15_bias_m_read_readvariableop1savev2_adam_dense_30_kernel_v_read_readvariableop/savev2_adam_dense_30_bias_v_read_readvariableop>savev2_adam_batch_normalization_21_gamma_v_read_readvariableop=savev2_adam_batch_normalization_21_beta_v_read_readvariableop1savev2_adam_dense_31_kernel_v_read_readvariableop/savev2_adam_dense_31_bias_v_read_readvariableop>savev2_adam_batch_normalization_22_gamma_v_read_readvariableop=savev2_adam_batch_normalization_22_beta_v_read_readvariableop<savev2_adam_conv2d_transpose_14_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_14_bias_v_read_readvariableop>savev2_adam_batch_normalization_23_gamma_v_read_readvariableop=savev2_adam_batch_normalization_23_beta_v_read_readvariableop<savev2_adam_conv2d_transpose_15_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_15_bias_v_read_readvariableop2savev2_adam_conv2d_14_kernel_m_read_readvariableop0savev2_adam_conv2d_14_bias_m_read_readvariableop2savev2_adam_conv2d_15_kernel_m_read_readvariableop0savev2_adam_conv2d_15_bias_m_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop/savev2_adam_dense_28_bias_m_read_readvariableop1savev2_adam_dense_29_kernel_m_read_readvariableop/savev2_adam_dense_29_bias_m_read_readvariableop2savev2_adam_conv2d_14_kernel_v_read_readvariableop0savev2_adam_conv2d_14_bias_v_read_readvariableop2savev2_adam_conv2d_15_kernel_v_read_readvariableop0savev2_adam_conv2d_15_bias_v_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableop/savev2_adam_dense_28_bias_v_read_readvariableop1savev2_adam_dense_29_kernel_v_read_readvariableop/savev2_adam_dense_29_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *g
dtypes]
[2Y		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :	d?:?:?:?:?:?:
??:?:?:?:?:?: @: : : : : : :: : : @:@:
??:?:	?:: : : : : : : : : : : :	d?:?:?:?:
??:?:?:?: @: : : : ::	d?:?:?:?:
??:?:?:?: @: : : : :: : : @:@:
??:?:	?:: : : @:@:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	d?:!

_output_shapes	
:?:!

_output_shapes	
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:,(
&
_output_shapes
: @: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:% !

_output_shapes
:	?: !

_output_shapes
::"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :%-!

_output_shapes
:	d?:!.

_output_shapes	
:?:!/

_output_shapes	
:?:!0

_output_shapes	
:?:&1"
 
_output_shapes
:
??:!2

_output_shapes	
:?:!3

_output_shapes	
:?:!4

_output_shapes	
:?:,5(
&
_output_shapes
: @: 6

_output_shapes
: : 7

_output_shapes
: : 8

_output_shapes
: :,9(
&
_output_shapes
: : :

_output_shapes
::%;!

_output_shapes
:	d?:!<

_output_shapes	
:?:!=

_output_shapes	
:?:!>

_output_shapes	
:?:&?"
 
_output_shapes
:
??:!@

_output_shapes	
:?:!A

_output_shapes	
:?:!B

_output_shapes	
:?:,C(
&
_output_shapes
: @: D

_output_shapes
: : E

_output_shapes
: : F

_output_shapes
: :,G(
&
_output_shapes
: : H

_output_shapes
::,I(
&
_output_shapes
: : J

_output_shapes
: :,K(
&
_output_shapes
: @: L

_output_shapes
:@:&M"
 
_output_shapes
:
??:!N

_output_shapes	
:?:%O!

_output_shapes
:	?: P

_output_shapes
::,Q(
&
_output_shapes
: : R

_output_shapes
: :,S(
&
_output_shapes
: @: T

_output_shapes
:@:&U"
 
_output_shapes
:
??:!V

_output_shapes	
:?:%W!

_output_shapes
:	?: X

_output_shapes
::Y

_output_shapes
: 
?
?
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1083457

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
/__inference_sequential_22_layer_call_fn_1083707
dense_30_input
unknown:	d?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?$

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_30_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_22_layer_call_and_return_conditional_losses_1083664w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????d
(
_user_specified_namedense_30_input
?	
?
E__inference_dense_30_layer_call_and_return_conditional_losses_1083560

inputs1
matmul_readvariableop_resource:	d?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
f
J__inference_activation_28_layer_call_and_return_conditional_losses_1085884

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:??????????[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_dense_29_layer_call_and_return_conditional_losses_1086376

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?"
J__inference_sequential_23_layer_call_and_return_conditional_losses_1085330

inputsH
5sequential_22_dense_30_matmul_readvariableop_resource:	d?E
6sequential_22_dense_30_biasadd_readvariableop_resource:	?[
Lsequential_22_batch_normalization_21_assignmovingavg_readvariableop_resource:	?]
Nsequential_22_batch_normalization_21_assignmovingavg_1_readvariableop_resource:	?Y
Jsequential_22_batch_normalization_21_batchnorm_mul_readvariableop_resource:	?U
Fsequential_22_batch_normalization_21_batchnorm_readvariableop_resource:	?I
5sequential_22_dense_31_matmul_readvariableop_resource:
??E
6sequential_22_dense_31_biasadd_readvariableop_resource:	?[
Lsequential_22_batch_normalization_22_assignmovingavg_readvariableop_resource:	?]
Nsequential_22_batch_normalization_22_assignmovingavg_1_readvariableop_resource:	?Y
Jsequential_22_batch_normalization_22_batchnorm_mul_readvariableop_resource:	?U
Fsequential_22_batch_normalization_22_batchnorm_readvariableop_resource:	?d
Jsequential_22_conv2d_transpose_14_conv2d_transpose_readvariableop_resource: @O
Asequential_22_conv2d_transpose_14_biasadd_readvariableop_resource: J
<sequential_22_batch_normalization_23_readvariableop_resource: L
>sequential_22_batch_normalization_23_readvariableop_1_resource: [
Msequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_resource: ]
Osequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource: d
Jsequential_22_conv2d_transpose_15_conv2d_transpose_readvariableop_resource: O
Asequential_22_conv2d_transpose_15_biasadd_readvariableop_resource:P
6sequential_21_conv2d_14_conv2d_readvariableop_resource: E
7sequential_21_conv2d_14_biasadd_readvariableop_resource: P
6sequential_21_conv2d_15_conv2d_readvariableop_resource: @E
7sequential_21_conv2d_15_biasadd_readvariableop_resource:@I
5sequential_21_dense_28_matmul_readvariableop_resource:
??E
6sequential_21_dense_28_biasadd_readvariableop_resource:	?H
5sequential_21_dense_29_matmul_readvariableop_resource:	?D
6sequential_21_dense_29_biasadd_readvariableop_resource:
identity??.sequential_21/conv2d_14/BiasAdd/ReadVariableOp?-sequential_21/conv2d_14/Conv2D/ReadVariableOp?.sequential_21/conv2d_15/BiasAdd/ReadVariableOp?-sequential_21/conv2d_15/Conv2D/ReadVariableOp?-sequential_21/dense_28/BiasAdd/ReadVariableOp?,sequential_21/dense_28/MatMul/ReadVariableOp?-sequential_21/dense_29/BiasAdd/ReadVariableOp?,sequential_21/dense_29/MatMul/ReadVariableOp?4sequential_22/batch_normalization_21/AssignMovingAvg?Csequential_22/batch_normalization_21/AssignMovingAvg/ReadVariableOp?6sequential_22/batch_normalization_21/AssignMovingAvg_1?Esequential_22/batch_normalization_21/AssignMovingAvg_1/ReadVariableOp?=sequential_22/batch_normalization_21/batchnorm/ReadVariableOp?Asequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOp?4sequential_22/batch_normalization_22/AssignMovingAvg?Csequential_22/batch_normalization_22/AssignMovingAvg/ReadVariableOp?6sequential_22/batch_normalization_22/AssignMovingAvg_1?Esequential_22/batch_normalization_22/AssignMovingAvg_1/ReadVariableOp?=sequential_22/batch_normalization_22/batchnorm/ReadVariableOp?Asequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOp?3sequential_22/batch_normalization_23/AssignNewValue?5sequential_22/batch_normalization_23/AssignNewValue_1?Dsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?Fsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?3sequential_22/batch_normalization_23/ReadVariableOp?5sequential_22/batch_normalization_23/ReadVariableOp_1?8sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOp?Asequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?8sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOp?Asequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?-sequential_22/dense_30/BiasAdd/ReadVariableOp?,sequential_22/dense_30/MatMul/ReadVariableOp?-sequential_22/dense_31/BiasAdd/ReadVariableOp?,sequential_22/dense_31/MatMul/ReadVariableOp?
,sequential_22/dense_30/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_30_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
sequential_22/dense_30/MatMulMatMulinputs4sequential_22/dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_22/dense_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_22/dense_30/BiasAddBiasAdd'sequential_22/dense_30/MatMul:product:05sequential_22/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
 sequential_22/activation_28/ReluRelu'sequential_22/dense_30/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
Csequential_22/batch_normalization_21/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
1sequential_22/batch_normalization_21/moments/meanMean.sequential_22/activation_28/Relu:activations:0Lsequential_22/batch_normalization_21/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
9sequential_22/batch_normalization_21/moments/StopGradientStopGradient:sequential_22/batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes
:	??
>sequential_22/batch_normalization_21/moments/SquaredDifferenceSquaredDifference.sequential_22/activation_28/Relu:activations:0Bsequential_22/batch_normalization_21/moments/StopGradient:output:0*
T0*(
_output_shapes
:???????????
Gsequential_22/batch_normalization_21/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
5sequential_22/batch_normalization_21/moments/varianceMeanBsequential_22/batch_normalization_21/moments/SquaredDifference:z:0Psequential_22/batch_normalization_21/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
4sequential_22/batch_normalization_21/moments/SqueezeSqueeze:sequential_22/batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 ?
6sequential_22/batch_normalization_21/moments/Squeeze_1Squeeze>sequential_22/batch_normalization_21/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 
:sequential_22/batch_normalization_21/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Csequential_22/batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOpLsequential_22_batch_normalization_21_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8sequential_22/batch_normalization_21/AssignMovingAvg/subSubKsequential_22/batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:0=sequential_22/batch_normalization_21/moments/Squeeze:output:0*
T0*
_output_shapes	
:??
8sequential_22/batch_normalization_21/AssignMovingAvg/mulMul<sequential_22/batch_normalization_21/AssignMovingAvg/sub:z:0Csequential_22/batch_normalization_21/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_21/AssignMovingAvgAssignSubVariableOpLsequential_22_batch_normalization_21_assignmovingavg_readvariableop_resource<sequential_22/batch_normalization_21/AssignMovingAvg/mul:z:0D^sequential_22/batch_normalization_21/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0?
<sequential_22/batch_normalization_21/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Esequential_22/batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOpNsequential_22_batch_normalization_21_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
:sequential_22/batch_normalization_21/AssignMovingAvg_1/subSubMsequential_22/batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:0?sequential_22/batch_normalization_21/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:??
:sequential_22/batch_normalization_21/AssignMovingAvg_1/mulMul>sequential_22/batch_normalization_21/AssignMovingAvg_1/sub:z:0Esequential_22/batch_normalization_21/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
6sequential_22/batch_normalization_21/AssignMovingAvg_1AssignSubVariableOpNsequential_22_batch_normalization_21_assignmovingavg_1_readvariableop_resource>sequential_22/batch_normalization_21/AssignMovingAvg_1/mul:z:0F^sequential_22/batch_normalization_21/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0y
4sequential_22/batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
2sequential_22/batch_normalization_21/batchnorm/addAddV2?sequential_22/batch_normalization_21/moments/Squeeze_1:output:0=sequential_22/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_21/batchnorm/RsqrtRsqrt6sequential_22/batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes	
:??
Asequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_22_batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2sequential_22/batch_normalization_21/batchnorm/mulMul8sequential_22/batch_normalization_21/batchnorm/Rsqrt:y:0Isequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_21/batchnorm/mul_1Mul.sequential_22/activation_28/Relu:activations:06sequential_22/batch_normalization_21/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
4sequential_22/batch_normalization_21/batchnorm/mul_2Mul=sequential_22/batch_normalization_21/moments/Squeeze:output:06sequential_22/batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
=sequential_22/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOpFsequential_22_batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2sequential_22/batch_normalization_21/batchnorm/subSubEsequential_22/batch_normalization_21/batchnorm/ReadVariableOp:value:08sequential_22/batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_21/batchnorm/add_1AddV28sequential_22/batch_normalization_21/batchnorm/mul_1:z:06sequential_22/batch_normalization_21/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
,sequential_22/dense_31/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_22/dense_31/MatMulMatMul8sequential_22/batch_normalization_21/batchnorm/add_1:z:04sequential_22/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_22/dense_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_22/dense_31/BiasAddBiasAdd'sequential_22/dense_31/MatMul:product:05sequential_22/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
 sequential_22/activation_29/ReluRelu'sequential_22/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
Csequential_22/batch_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
1sequential_22/batch_normalization_22/moments/meanMean.sequential_22/activation_29/Relu:activations:0Lsequential_22/batch_normalization_22/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
9sequential_22/batch_normalization_22/moments/StopGradientStopGradient:sequential_22/batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes
:	??
>sequential_22/batch_normalization_22/moments/SquaredDifferenceSquaredDifference.sequential_22/activation_29/Relu:activations:0Bsequential_22/batch_normalization_22/moments/StopGradient:output:0*
T0*(
_output_shapes
:???????????
Gsequential_22/batch_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
5sequential_22/batch_normalization_22/moments/varianceMeanBsequential_22/batch_normalization_22/moments/SquaredDifference:z:0Psequential_22/batch_normalization_22/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
4sequential_22/batch_normalization_22/moments/SqueezeSqueeze:sequential_22/batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 ?
6sequential_22/batch_normalization_22/moments/Squeeze_1Squeeze>sequential_22/batch_normalization_22/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 
:sequential_22/batch_normalization_22/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Csequential_22/batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOpLsequential_22_batch_normalization_22_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8sequential_22/batch_normalization_22/AssignMovingAvg/subSubKsequential_22/batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:0=sequential_22/batch_normalization_22/moments/Squeeze:output:0*
T0*
_output_shapes	
:??
8sequential_22/batch_normalization_22/AssignMovingAvg/mulMul<sequential_22/batch_normalization_22/AssignMovingAvg/sub:z:0Csequential_22/batch_normalization_22/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_22/AssignMovingAvgAssignSubVariableOpLsequential_22_batch_normalization_22_assignmovingavg_readvariableop_resource<sequential_22/batch_normalization_22/AssignMovingAvg/mul:z:0D^sequential_22/batch_normalization_22/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0?
<sequential_22/batch_normalization_22/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
Esequential_22/batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOpNsequential_22_batch_normalization_22_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
:sequential_22/batch_normalization_22/AssignMovingAvg_1/subSubMsequential_22/batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:0?sequential_22/batch_normalization_22/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:??
:sequential_22/batch_normalization_22/AssignMovingAvg_1/mulMul>sequential_22/batch_normalization_22/AssignMovingAvg_1/sub:z:0Esequential_22/batch_normalization_22/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
6sequential_22/batch_normalization_22/AssignMovingAvg_1AssignSubVariableOpNsequential_22_batch_normalization_22_assignmovingavg_1_readvariableop_resource>sequential_22/batch_normalization_22/AssignMovingAvg_1/mul:z:0F^sequential_22/batch_normalization_22/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0y
4sequential_22/batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
2sequential_22/batch_normalization_22/batchnorm/addAddV2?sequential_22/batch_normalization_22/moments/Squeeze_1:output:0=sequential_22/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_22/batchnorm/RsqrtRsqrt6sequential_22/batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:??
Asequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_22_batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2sequential_22/batch_normalization_22/batchnorm/mulMul8sequential_22/batch_normalization_22/batchnorm/Rsqrt:y:0Isequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_22/batchnorm/mul_1Mul.sequential_22/activation_29/Relu:activations:06sequential_22/batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
4sequential_22/batch_normalization_22/batchnorm/mul_2Mul=sequential_22/batch_normalization_22/moments/Squeeze:output:06sequential_22/batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
=sequential_22/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOpFsequential_22_batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
2sequential_22/batch_normalization_22/batchnorm/subSubEsequential_22/batch_normalization_22/batchnorm/ReadVariableOp:value:08sequential_22/batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
4sequential_22/batch_normalization_22/batchnorm/add_1AddV28sequential_22/batch_normalization_22/batchnorm/mul_1:z:06sequential_22/batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
sequential_22/reshape_7/ShapeShape8sequential_22/batch_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:u
+sequential_22/reshape_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_22/reshape_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_22/reshape_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%sequential_22/reshape_7/strided_sliceStridedSlice&sequential_22/reshape_7/Shape:output:04sequential_22/reshape_7/strided_slice/stack:output:06sequential_22/reshape_7/strided_slice/stack_1:output:06sequential_22/reshape_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'sequential_22/reshape_7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :i
'sequential_22/reshape_7/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :i
'sequential_22/reshape_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
%sequential_22/reshape_7/Reshape/shapePack.sequential_22/reshape_7/strided_slice:output:00sequential_22/reshape_7/Reshape/shape/1:output:00sequential_22/reshape_7/Reshape/shape/2:output:00sequential_22/reshape_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
sequential_22/reshape_7/ReshapeReshape8sequential_22/batch_normalization_22/batchnorm/add_1:z:0.sequential_22/reshape_7/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@
'sequential_22/conv2d_transpose_14/ShapeShape(sequential_22/reshape_7/Reshape:output:0*
T0*
_output_shapes
:
5sequential_22/conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7sequential_22/conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential_22/conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_22/conv2d_transpose_14/strided_sliceStridedSlice0sequential_22/conv2d_transpose_14/Shape:output:0>sequential_22/conv2d_transpose_14/strided_slice/stack:output:0@sequential_22/conv2d_transpose_14/strided_slice/stack_1:output:0@sequential_22/conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_22/conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_22/conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_22/conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_22/conv2d_transpose_14/stackPack8sequential_22/conv2d_transpose_14/strided_slice:output:02sequential_22/conv2d_transpose_14/stack/1:output:02sequential_22/conv2d_transpose_14/stack/2:output:02sequential_22/conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:?
7sequential_22/conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9sequential_22/conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential_22/conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1sequential_22/conv2d_transpose_14/strided_slice_1StridedSlice0sequential_22/conv2d_transpose_14/stack:output:0@sequential_22/conv2d_transpose_14/strided_slice_1/stack:output:0Bsequential_22/conv2d_transpose_14/strided_slice_1/stack_1:output:0Bsequential_22/conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Asequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_22_conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
2sequential_22/conv2d_transpose_14/conv2d_transposeConv2DBackpropInput0sequential_22/conv2d_transpose_14/stack:output:0Isequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0(sequential_22/reshape_7/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
8sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOpAsequential_22_conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
)sequential_22/conv2d_transpose_14/BiasAddBiasAdd;sequential_22/conv2d_transpose_14/conv2d_transpose:output:0@sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
 sequential_22/activation_30/ReluRelu2sequential_22/conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
3sequential_22/batch_normalization_23/ReadVariableOpReadVariableOp<sequential_22_batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype0?
5sequential_22/batch_normalization_23/ReadVariableOp_1ReadVariableOp>sequential_22_batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Dsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Fsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5sequential_22/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3.sequential_22/activation_30/Relu:activations:0;sequential_22/batch_normalization_23/ReadVariableOp:value:0=sequential_22/batch_normalization_23/ReadVariableOp_1:value:0Lsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_22/batch_normalization_23/AssignNewValueAssignVariableOpMsequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_resourceBsequential_22/batch_normalization_23/FusedBatchNormV3:batch_mean:0E^sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_22/batch_normalization_23/AssignNewValue_1AssignVariableOpOsequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resourceFsequential_22/batch_normalization_23/FusedBatchNormV3:batch_variance:0G^sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
'sequential_22/conv2d_transpose_15/ShapeShape9sequential_22/batch_normalization_23/FusedBatchNormV3:y:0*
T0*
_output_shapes
:
5sequential_22/conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7sequential_22/conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential_22/conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_22/conv2d_transpose_15/strided_sliceStridedSlice0sequential_22/conv2d_transpose_15/Shape:output:0>sequential_22/conv2d_transpose_15/strided_slice/stack:output:0@sequential_22/conv2d_transpose_15/strided_slice/stack_1:output:0@sequential_22/conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_22/conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_22/conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :k
)sequential_22/conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
'sequential_22/conv2d_transpose_15/stackPack8sequential_22/conv2d_transpose_15/strided_slice:output:02sequential_22/conv2d_transpose_15/stack/1:output:02sequential_22/conv2d_transpose_15/stack/2:output:02sequential_22/conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:?
7sequential_22/conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9sequential_22/conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential_22/conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1sequential_22/conv2d_transpose_15/strided_slice_1StridedSlice0sequential_22/conv2d_transpose_15/stack:output:0@sequential_22/conv2d_transpose_15/strided_slice_1/stack:output:0Bsequential_22/conv2d_transpose_15/strided_slice_1/stack_1:output:0Bsequential_22/conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Asequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_22_conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
2sequential_22/conv2d_transpose_15/conv2d_transposeConv2DBackpropInput0sequential_22/conv2d_transpose_15/stack:output:0Isequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:09sequential_22/batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
8sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOpAsequential_22_conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)sequential_22/conv2d_transpose_15/BiasAddBiasAdd;sequential_22/conv2d_transpose_15/conv2d_transpose:output:0@sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
 sequential_22/activation_31/TanhTanh2sequential_22/conv2d_transpose_15/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
-sequential_21/conv2d_14/Conv2D/ReadVariableOpReadVariableOp6sequential_21_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential_21/conv2d_14/Conv2DConv2D$sequential_22/activation_31/Tanh:y:05sequential_21/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
.sequential_21/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp7sequential_21_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_21/conv2d_14/BiasAddBiasAdd'sequential_21/conv2d_14/Conv2D:output:06sequential_21/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
&sequential_21/leaky_re_lu_21/LeakyRelu	LeakyRelu(sequential_21/conv2d_14/BiasAdd:output:0*/
_output_shapes
:????????? ?
-sequential_21/conv2d_15/Conv2D/ReadVariableOpReadVariableOp6sequential_21_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
sequential_21/conv2d_15/Conv2DConv2D4sequential_21/leaky_re_lu_21/LeakyRelu:activations:05sequential_21/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
.sequential_21/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp7sequential_21_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_21/conv2d_15/BiasAddBiasAdd'sequential_21/conv2d_15/Conv2D:output:06sequential_21/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
&sequential_21/leaky_re_lu_22/LeakyRelu	LeakyRelu(sequential_21/conv2d_15/BiasAdd:output:0*/
_output_shapes
:?????????@n
sequential_21/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
sequential_21/flatten_7/ReshapeReshape4sequential_21/leaky_re_lu_22/LeakyRelu:activations:0&sequential_21/flatten_7/Const:output:0*
T0*(
_output_shapes
:???????????
,sequential_21/dense_28/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_21/dense_28/MatMulMatMul(sequential_21/flatten_7/Reshape:output:04sequential_21/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_21/dense_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_21/dense_28/BiasAddBiasAdd'sequential_21/dense_28/MatMul:product:05sequential_21/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&sequential_21/leaky_re_lu_23/LeakyRelu	LeakyRelu'sequential_21/dense_28/BiasAdd:output:0*(
_output_shapes
:???????????
,sequential_21/dense_29/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential_21/dense_29/MatMulMatMul4sequential_21/leaky_re_lu_23/LeakyRelu:activations:04sequential_21/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-sequential_21/dense_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_21/dense_29/BiasAddBiasAdd'sequential_21/dense_29/MatMul:product:05sequential_21/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_21/dense_29/SigmoidSigmoid'sequential_21/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:?????????q
IdentityIdentity"sequential_21/dense_29/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^sequential_21/conv2d_14/BiasAdd/ReadVariableOp.^sequential_21/conv2d_14/Conv2D/ReadVariableOp/^sequential_21/conv2d_15/BiasAdd/ReadVariableOp.^sequential_21/conv2d_15/Conv2D/ReadVariableOp.^sequential_21/dense_28/BiasAdd/ReadVariableOp-^sequential_21/dense_28/MatMul/ReadVariableOp.^sequential_21/dense_29/BiasAdd/ReadVariableOp-^sequential_21/dense_29/MatMul/ReadVariableOp5^sequential_22/batch_normalization_21/AssignMovingAvgD^sequential_22/batch_normalization_21/AssignMovingAvg/ReadVariableOp7^sequential_22/batch_normalization_21/AssignMovingAvg_1F^sequential_22/batch_normalization_21/AssignMovingAvg_1/ReadVariableOp>^sequential_22/batch_normalization_21/batchnorm/ReadVariableOpB^sequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOp5^sequential_22/batch_normalization_22/AssignMovingAvgD^sequential_22/batch_normalization_22/AssignMovingAvg/ReadVariableOp7^sequential_22/batch_normalization_22/AssignMovingAvg_1F^sequential_22/batch_normalization_22/AssignMovingAvg_1/ReadVariableOp>^sequential_22/batch_normalization_22/batchnorm/ReadVariableOpB^sequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOp4^sequential_22/batch_normalization_23/AssignNewValue6^sequential_22/batch_normalization_23/AssignNewValue_1E^sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOpG^sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_14^sequential_22/batch_normalization_23/ReadVariableOp6^sequential_22/batch_normalization_23/ReadVariableOp_19^sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOpB^sequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOp9^sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOpB^sequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOp.^sequential_22/dense_30/BiasAdd/ReadVariableOp-^sequential_22/dense_30/MatMul/ReadVariableOp.^sequential_22/dense_31/BiasAdd/ReadVariableOp-^sequential_22/dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.sequential_21/conv2d_14/BiasAdd/ReadVariableOp.sequential_21/conv2d_14/BiasAdd/ReadVariableOp2^
-sequential_21/conv2d_14/Conv2D/ReadVariableOp-sequential_21/conv2d_14/Conv2D/ReadVariableOp2`
.sequential_21/conv2d_15/BiasAdd/ReadVariableOp.sequential_21/conv2d_15/BiasAdd/ReadVariableOp2^
-sequential_21/conv2d_15/Conv2D/ReadVariableOp-sequential_21/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_21/dense_28/BiasAdd/ReadVariableOp-sequential_21/dense_28/BiasAdd/ReadVariableOp2\
,sequential_21/dense_28/MatMul/ReadVariableOp,sequential_21/dense_28/MatMul/ReadVariableOp2^
-sequential_21/dense_29/BiasAdd/ReadVariableOp-sequential_21/dense_29/BiasAdd/ReadVariableOp2\
,sequential_21/dense_29/MatMul/ReadVariableOp,sequential_21/dense_29/MatMul/ReadVariableOp2l
4sequential_22/batch_normalization_21/AssignMovingAvg4sequential_22/batch_normalization_21/AssignMovingAvg2?
Csequential_22/batch_normalization_21/AssignMovingAvg/ReadVariableOpCsequential_22/batch_normalization_21/AssignMovingAvg/ReadVariableOp2p
6sequential_22/batch_normalization_21/AssignMovingAvg_16sequential_22/batch_normalization_21/AssignMovingAvg_12?
Esequential_22/batch_normalization_21/AssignMovingAvg_1/ReadVariableOpEsequential_22/batch_normalization_21/AssignMovingAvg_1/ReadVariableOp2~
=sequential_22/batch_normalization_21/batchnorm/ReadVariableOp=sequential_22/batch_normalization_21/batchnorm/ReadVariableOp2?
Asequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOpAsequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOp2l
4sequential_22/batch_normalization_22/AssignMovingAvg4sequential_22/batch_normalization_22/AssignMovingAvg2?
Csequential_22/batch_normalization_22/AssignMovingAvg/ReadVariableOpCsequential_22/batch_normalization_22/AssignMovingAvg/ReadVariableOp2p
6sequential_22/batch_normalization_22/AssignMovingAvg_16sequential_22/batch_normalization_22/AssignMovingAvg_12?
Esequential_22/batch_normalization_22/AssignMovingAvg_1/ReadVariableOpEsequential_22/batch_normalization_22/AssignMovingAvg_1/ReadVariableOp2~
=sequential_22/batch_normalization_22/batchnorm/ReadVariableOp=sequential_22/batch_normalization_22/batchnorm/ReadVariableOp2?
Asequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOpAsequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOp2j
3sequential_22/batch_normalization_23/AssignNewValue3sequential_22/batch_normalization_23/AssignNewValue2n
5sequential_22/batch_normalization_23/AssignNewValue_15sequential_22/batch_normalization_23/AssignNewValue_12?
Dsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOpDsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2?
Fsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Fsequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12j
3sequential_22/batch_normalization_23/ReadVariableOp3sequential_22/batch_normalization_23/ReadVariableOp2n
5sequential_22/batch_normalization_23/ReadVariableOp_15sequential_22/batch_normalization_23/ReadVariableOp_12t
8sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOp8sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOp2?
Asequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOpAsequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOp2t
8sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOp8sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOp2?
Asequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOpAsequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOp2^
-sequential_22/dense_30/BiasAdd/ReadVariableOp-sequential_22/dense_30/BiasAdd/ReadVariableOp2\
,sequential_22/dense_30/MatMul/ReadVariableOp,sequential_22/dense_30/MatMul/ReadVariableOp2^
-sequential_22/dense_31/BiasAdd/ReadVariableOp-sequential_22/dense_31/BiasAdd/ReadVariableOp2\
,sequential_22/dense_31/MatMul/ReadVariableOp,sequential_22/dense_31/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?#
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084392
conv2d_14_input+
conv2d_14_1084367: 
conv2d_14_1084369: +
conv2d_15_1084373: @
conv2d_15_1084375:@$
dense_28_1084380:
??
dense_28_1084382:	?#
dense_29_1084386:	?
dense_29_1084388:
identity??!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallconv2d_14_inputconv2d_14_1084367conv2d_14_1084369*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_14_layer_call_and_return_conditional_losses_1084077?
leaky_re_lu_21/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1084088?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0conv2d_15_1084373conv2d_15_1084375*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1084100?
leaky_re_lu_22/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1084111?
flatten_7/PartitionedCallPartitionedCall'leaky_re_lu_22/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_1084119?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_28_1084380dense_28_1084382*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_1084131?
leaky_re_lu_23/PartitionedCallPartitionedCall)dense_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1084142?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0dense_29_1084386dense_29_1084388*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_1084155x
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_14_input
?=
?

J__inference_sequential_22_layer_call_and_return_conditional_losses_1084004
dense_30_input#
dense_30_1083951:	d?
dense_30_1083953:	?-
batch_normalization_21_1083957:	?-
batch_normalization_21_1083959:	?-
batch_normalization_21_1083961:	?-
batch_normalization_21_1083963:	?$
dense_31_1083966:
??
dense_31_1083968:	?-
batch_normalization_22_1083972:	?-
batch_normalization_22_1083974:	?-
batch_normalization_22_1083976:	?-
batch_normalization_22_1083978:	?5
conv2d_transpose_14_1083982: @)
conv2d_transpose_14_1083984: ,
batch_normalization_23_1083988: ,
batch_normalization_23_1083990: ,
batch_normalization_23_1083992: ,
batch_normalization_23_1083994: 5
conv2d_transpose_15_1083997: )
conv2d_transpose_15_1083999:
identity??.batch_normalization_21/StatefulPartitionedCall?.batch_normalization_22/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCalldense_30_inputdense_30_1083951dense_30_1083953*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_30_layer_call_and_return_conditional_losses_1083560?
activation_28/PartitionedCallPartitionedCall)dense_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_28_layer_call_and_return_conditional_losses_1083571?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall&activation_28/PartitionedCall:output:0batch_normalization_21_1083957batch_normalization_21_1083959batch_normalization_21_1083961batch_normalization_21_1083963*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1083251?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0dense_31_1083966dense_31_1083968*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_1083592?
activation_29/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_29_layer_call_and_return_conditional_losses_1083603?
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall&activation_29/PartitionedCall:output:0batch_normalization_22_1083972batch_normalization_22_1083974batch_normalization_22_1083976batch_normalization_22_1083978*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1083333?
reshape_7/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_reshape_7_layer_call_and_return_conditional_losses_1083628?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv2d_transpose_14_1083982conv2d_transpose_14_1083984*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_1083428?
activation_30/PartitionedCallPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_30_layer_call_and_return_conditional_losses_1083640?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall&activation_30/PartitionedCall:output:0batch_normalization_23_1083988batch_normalization_23_1083990batch_normalization_23_1083992batch_normalization_23_1083994*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1083457?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0conv2d_transpose_15_1083997conv2d_transpose_15_1083999*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_1083536?
activation_31/PartitionedCallPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_31_layer_call_and_return_conditional_losses_1083661}
IdentityIdentity&activation_31/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:W S
'
_output_shapes
:?????????d
(
_user_specified_namedense_30_input
?
?
*__inference_dense_28_layer_call_fn_1086336

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_1084131p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_dense_29_layer_call_and_return_conditional_losses_1084155

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_dense_29_layer_call_fn_1086365

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_1084155o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_activation_30_layer_call_and_return_conditional_losses_1083640

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1083333

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?

J__inference_sequential_23_layer_call_and_return_conditional_losses_1084458

inputs(
sequential_22_1084399:	d?$
sequential_22_1084401:	?$
sequential_22_1084403:	?$
sequential_22_1084405:	?$
sequential_22_1084407:	?$
sequential_22_1084409:	?)
sequential_22_1084411:
??$
sequential_22_1084413:	?$
sequential_22_1084415:	?$
sequential_22_1084417:	?$
sequential_22_1084419:	?$
sequential_22_1084421:	?/
sequential_22_1084423: @#
sequential_22_1084425: #
sequential_22_1084427: #
sequential_22_1084429: #
sequential_22_1084431: #
sequential_22_1084433: /
sequential_22_1084435: #
sequential_22_1084437:/
sequential_21_1084440: #
sequential_21_1084442: /
sequential_21_1084444: @#
sequential_21_1084446:@)
sequential_21_1084448:
??$
sequential_21_1084450:	?(
sequential_21_1084452:	?#
sequential_21_1084454:
identity??%sequential_21/StatefulPartitionedCall?%sequential_22/StatefulPartitionedCall?
%sequential_22/StatefulPartitionedCallStatefulPartitionedCallinputssequential_22_1084399sequential_22_1084401sequential_22_1084403sequential_22_1084405sequential_22_1084407sequential_22_1084409sequential_22_1084411sequential_22_1084413sequential_22_1084415sequential_22_1084417sequential_22_1084419sequential_22_1084421sequential_22_1084423sequential_22_1084425sequential_22_1084427sequential_22_1084429sequential_22_1084431sequential_22_1084433sequential_22_1084435sequential_22_1084437* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_22_layer_call_and_return_conditional_losses_1083664?
%sequential_21/StatefulPartitionedCallStatefulPartitionedCall.sequential_22/StatefulPartitionedCall:output:0sequential_21_1084440sequential_21_1084442sequential_21_1084444sequential_21_1084446sequential_21_1084448sequential_21_1084450sequential_21_1084452sequential_21_1084454*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084162}
IdentityIdentity.sequential_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^sequential_21/StatefulPartitionedCall&^sequential_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%sequential_21/StatefulPartitionedCall%sequential_21/StatefulPartitionedCall2N
%sequential_22/StatefulPartitionedCall%sequential_22/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
/__inference_sequential_22_layer_call_fn_1085438

inputs
unknown:	d?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?$

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_22_layer_call_and_return_conditional_losses_1083664w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?%
"__inference__wrapped_model_1083227
sequential_22_inputV
Csequential_23_sequential_22_dense_30_matmul_readvariableop_resource:	d?S
Dsequential_23_sequential_22_dense_30_biasadd_readvariableop_resource:	?c
Tsequential_23_sequential_22_batch_normalization_21_batchnorm_readvariableop_resource:	?g
Xsequential_23_sequential_22_batch_normalization_21_batchnorm_mul_readvariableop_resource:	?e
Vsequential_23_sequential_22_batch_normalization_21_batchnorm_readvariableop_1_resource:	?e
Vsequential_23_sequential_22_batch_normalization_21_batchnorm_readvariableop_2_resource:	?W
Csequential_23_sequential_22_dense_31_matmul_readvariableop_resource:
??S
Dsequential_23_sequential_22_dense_31_biasadd_readvariableop_resource:	?c
Tsequential_23_sequential_22_batch_normalization_22_batchnorm_readvariableop_resource:	?g
Xsequential_23_sequential_22_batch_normalization_22_batchnorm_mul_readvariableop_resource:	?e
Vsequential_23_sequential_22_batch_normalization_22_batchnorm_readvariableop_1_resource:	?e
Vsequential_23_sequential_22_batch_normalization_22_batchnorm_readvariableop_2_resource:	?r
Xsequential_23_sequential_22_conv2d_transpose_14_conv2d_transpose_readvariableop_resource: @]
Osequential_23_sequential_22_conv2d_transpose_14_biasadd_readvariableop_resource: X
Jsequential_23_sequential_22_batch_normalization_23_readvariableop_resource: Z
Lsequential_23_sequential_22_batch_normalization_23_readvariableop_1_resource: i
[sequential_23_sequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_resource: k
]sequential_23_sequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource: r
Xsequential_23_sequential_22_conv2d_transpose_15_conv2d_transpose_readvariableop_resource: ]
Osequential_23_sequential_22_conv2d_transpose_15_biasadd_readvariableop_resource:^
Dsequential_23_sequential_21_conv2d_14_conv2d_readvariableop_resource: S
Esequential_23_sequential_21_conv2d_14_biasadd_readvariableop_resource: ^
Dsequential_23_sequential_21_conv2d_15_conv2d_readvariableop_resource: @S
Esequential_23_sequential_21_conv2d_15_biasadd_readvariableop_resource:@W
Csequential_23_sequential_21_dense_28_matmul_readvariableop_resource:
??S
Dsequential_23_sequential_21_dense_28_biasadd_readvariableop_resource:	?V
Csequential_23_sequential_21_dense_29_matmul_readvariableop_resource:	?R
Dsequential_23_sequential_21_dense_29_biasadd_readvariableop_resource:
identity??<sequential_23/sequential_21/conv2d_14/BiasAdd/ReadVariableOp?;sequential_23/sequential_21/conv2d_14/Conv2D/ReadVariableOp?<sequential_23/sequential_21/conv2d_15/BiasAdd/ReadVariableOp?;sequential_23/sequential_21/conv2d_15/Conv2D/ReadVariableOp?;sequential_23/sequential_21/dense_28/BiasAdd/ReadVariableOp?:sequential_23/sequential_21/dense_28/MatMul/ReadVariableOp?;sequential_23/sequential_21/dense_29/BiasAdd/ReadVariableOp?:sequential_23/sequential_21/dense_29/MatMul/ReadVariableOp?Ksequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp?Msequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_1?Msequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_2?Osequential_23/sequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOp?Ksequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp?Msequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_1?Msequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_2?Osequential_23/sequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOp?Rsequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp?Tsequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1?Asequential_23/sequential_22/batch_normalization_23/ReadVariableOp?Csequential_23/sequential_22/batch_normalization_23/ReadVariableOp_1?Fsequential_23/sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOp?Osequential_23/sequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?Fsequential_23/sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOp?Osequential_23/sequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?;sequential_23/sequential_22/dense_30/BiasAdd/ReadVariableOp?:sequential_23/sequential_22/dense_30/MatMul/ReadVariableOp?;sequential_23/sequential_22/dense_31/BiasAdd/ReadVariableOp?:sequential_23/sequential_22/dense_31/MatMul/ReadVariableOp?
:sequential_23/sequential_22/dense_30/MatMul/ReadVariableOpReadVariableOpCsequential_23_sequential_22_dense_30_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
+sequential_23/sequential_22/dense_30/MatMulMatMulsequential_22_inputBsequential_23/sequential_22/dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
;sequential_23/sequential_22/dense_30/BiasAdd/ReadVariableOpReadVariableOpDsequential_23_sequential_22_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,sequential_23/sequential_22/dense_30/BiasAddBiasAdd5sequential_23/sequential_22/dense_30/MatMul:product:0Csequential_23/sequential_22/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
.sequential_23/sequential_22/activation_28/ReluRelu5sequential_23/sequential_22/dense_30/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
Ksequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOpTsequential_23_sequential_22_batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Bsequential_23/sequential_22/batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
@sequential_23/sequential_22/batch_normalization_21/batchnorm/addAddV2Ssequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp:value:0Ksequential_23/sequential_22/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes	
:??
Bsequential_23/sequential_22/batch_normalization_21/batchnorm/RsqrtRsqrtDsequential_23/sequential_22/batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes	
:??
Osequential_23/sequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOpXsequential_23_sequential_22_batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
@sequential_23/sequential_22/batch_normalization_21/batchnorm/mulMulFsequential_23/sequential_22/batch_normalization_21/batchnorm/Rsqrt:y:0Wsequential_23/sequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
Bsequential_23/sequential_22/batch_normalization_21/batchnorm/mul_1Mul<sequential_23/sequential_22/activation_28/Relu:activations:0Dsequential_23/sequential_22/batch_normalization_21/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
Msequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOpVsequential_23_sequential_22_batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Bsequential_23/sequential_22/batch_normalization_21/batchnorm/mul_2MulUsequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_1:value:0Dsequential_23/sequential_22/batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
Msequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOpVsequential_23_sequential_22_batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
@sequential_23/sequential_22/batch_normalization_21/batchnorm/subSubUsequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_2:value:0Fsequential_23/sequential_22/batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
Bsequential_23/sequential_22/batch_normalization_21/batchnorm/add_1AddV2Fsequential_23/sequential_22/batch_normalization_21/batchnorm/mul_1:z:0Dsequential_23/sequential_22/batch_normalization_21/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
:sequential_23/sequential_22/dense_31/MatMul/ReadVariableOpReadVariableOpCsequential_23_sequential_22_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
+sequential_23/sequential_22/dense_31/MatMulMatMulFsequential_23/sequential_22/batch_normalization_21/batchnorm/add_1:z:0Bsequential_23/sequential_22/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
;sequential_23/sequential_22/dense_31/BiasAdd/ReadVariableOpReadVariableOpDsequential_23_sequential_22_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,sequential_23/sequential_22/dense_31/BiasAddBiasAdd5sequential_23/sequential_22/dense_31/MatMul:product:0Csequential_23/sequential_22/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
.sequential_23/sequential_22/activation_29/ReluRelu5sequential_23/sequential_22/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
Ksequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOpTsequential_23_sequential_22_batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Bsequential_23/sequential_22/batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
@sequential_23/sequential_22/batch_normalization_22/batchnorm/addAddV2Ssequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp:value:0Ksequential_23/sequential_22/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:??
Bsequential_23/sequential_22/batch_normalization_22/batchnorm/RsqrtRsqrtDsequential_23/sequential_22/batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:??
Osequential_23/sequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOpXsequential_23_sequential_22_batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
@sequential_23/sequential_22/batch_normalization_22/batchnorm/mulMulFsequential_23/sequential_22/batch_normalization_22/batchnorm/Rsqrt:y:0Wsequential_23/sequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
Bsequential_23/sequential_22/batch_normalization_22/batchnorm/mul_1Mul<sequential_23/sequential_22/activation_29/Relu:activations:0Dsequential_23/sequential_22/batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
Msequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOpVsequential_23_sequential_22_batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Bsequential_23/sequential_22/batch_normalization_22/batchnorm/mul_2MulUsequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_1:value:0Dsequential_23/sequential_22/batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
Msequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOpVsequential_23_sequential_22_batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
@sequential_23/sequential_22/batch_normalization_22/batchnorm/subSubUsequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_2:value:0Fsequential_23/sequential_22/batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
Bsequential_23/sequential_22/batch_normalization_22/batchnorm/add_1AddV2Fsequential_23/sequential_22/batch_normalization_22/batchnorm/mul_1:z:0Dsequential_23/sequential_22/batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
+sequential_23/sequential_22/reshape_7/ShapeShapeFsequential_23/sequential_22/batch_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:?
9sequential_23/sequential_22/reshape_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;sequential_23/sequential_22/reshape_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;sequential_23/sequential_22/reshape_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3sequential_23/sequential_22/reshape_7/strided_sliceStridedSlice4sequential_23/sequential_22/reshape_7/Shape:output:0Bsequential_23/sequential_22/reshape_7/strided_slice/stack:output:0Dsequential_23/sequential_22/reshape_7/strided_slice/stack_1:output:0Dsequential_23/sequential_22/reshape_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5sequential_23/sequential_22/reshape_7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :w
5sequential_23/sequential_22/reshape_7/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :w
5sequential_23/sequential_22/reshape_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
3sequential_23/sequential_22/reshape_7/Reshape/shapePack<sequential_23/sequential_22/reshape_7/strided_slice:output:0>sequential_23/sequential_22/reshape_7/Reshape/shape/1:output:0>sequential_23/sequential_22/reshape_7/Reshape/shape/2:output:0>sequential_23/sequential_22/reshape_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
-sequential_23/sequential_22/reshape_7/ReshapeReshapeFsequential_23/sequential_22/batch_normalization_22/batchnorm/add_1:z:0<sequential_23/sequential_22/reshape_7/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@?
5sequential_23/sequential_22/conv2d_transpose_14/ShapeShape6sequential_23/sequential_22/reshape_7/Reshape:output:0*
T0*
_output_shapes
:?
Csequential_23/sequential_22/conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esequential_23/sequential_22/conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential_23/sequential_22/conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=sequential_23/sequential_22/conv2d_transpose_14/strided_sliceStridedSlice>sequential_23/sequential_22/conv2d_transpose_14/Shape:output:0Lsequential_23/sequential_22/conv2d_transpose_14/strided_slice/stack:output:0Nsequential_23/sequential_22/conv2d_transpose_14/strided_slice/stack_1:output:0Nsequential_23/sequential_22/conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
7sequential_23/sequential_22/conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :y
7sequential_23/sequential_22/conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :y
7sequential_23/sequential_22/conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
5sequential_23/sequential_22/conv2d_transpose_14/stackPackFsequential_23/sequential_22/conv2d_transpose_14/strided_slice:output:0@sequential_23/sequential_22/conv2d_transpose_14/stack/1:output:0@sequential_23/sequential_22/conv2d_transpose_14/stack/2:output:0@sequential_23/sequential_22/conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:?
Esequential_23/sequential_22/conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Gsequential_23/sequential_22/conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Gsequential_23/sequential_22/conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential_23/sequential_22/conv2d_transpose_14/strided_slice_1StridedSlice>sequential_23/sequential_22/conv2d_transpose_14/stack:output:0Nsequential_23/sequential_22/conv2d_transpose_14/strided_slice_1/stack:output:0Psequential_23/sequential_22/conv2d_transpose_14/strided_slice_1/stack_1:output:0Psequential_23/sequential_22/conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Osequential_23/sequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOpXsequential_23_sequential_22_conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
@sequential_23/sequential_22/conv2d_transpose_14/conv2d_transposeConv2DBackpropInput>sequential_23/sequential_22/conv2d_transpose_14/stack:output:0Wsequential_23/sequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:06sequential_23/sequential_22/reshape_7/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
Fsequential_23/sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOpOsequential_23_sequential_22_conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
7sequential_23/sequential_22/conv2d_transpose_14/BiasAddBiasAddIsequential_23/sequential_22/conv2d_transpose_14/conv2d_transpose:output:0Nsequential_23/sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
.sequential_23/sequential_22/activation_30/ReluRelu@sequential_23/sequential_22/conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
Asequential_23/sequential_22/batch_normalization_23/ReadVariableOpReadVariableOpJsequential_23_sequential_22_batch_normalization_23_readvariableop_resource*
_output_shapes
: *
dtype0?
Csequential_23/sequential_22/batch_normalization_23/ReadVariableOp_1ReadVariableOpLsequential_23_sequential_22_batch_normalization_23_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Rsequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp[sequential_23_sequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Tsequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]sequential_23_sequential_22_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Csequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3<sequential_23/sequential_22/activation_30/Relu:activations:0Isequential_23/sequential_22/batch_normalization_23/ReadVariableOp:value:0Ksequential_23/sequential_22/batch_normalization_23/ReadVariableOp_1:value:0Zsequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0\sequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
5sequential_23/sequential_22/conv2d_transpose_15/ShapeShapeGsequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3:y:0*
T0*
_output_shapes
:?
Csequential_23/sequential_22/conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Esequential_23/sequential_22/conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Esequential_23/sequential_22/conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
=sequential_23/sequential_22/conv2d_transpose_15/strided_sliceStridedSlice>sequential_23/sequential_22/conv2d_transpose_15/Shape:output:0Lsequential_23/sequential_22/conv2d_transpose_15/strided_slice/stack:output:0Nsequential_23/sequential_22/conv2d_transpose_15/strided_slice/stack_1:output:0Nsequential_23/sequential_22/conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
7sequential_23/sequential_22/conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :y
7sequential_23/sequential_22/conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :y
7sequential_23/sequential_22/conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
5sequential_23/sequential_22/conv2d_transpose_15/stackPackFsequential_23/sequential_22/conv2d_transpose_15/strided_slice:output:0@sequential_23/sequential_22/conv2d_transpose_15/stack/1:output:0@sequential_23/sequential_22/conv2d_transpose_15/stack/2:output:0@sequential_23/sequential_22/conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:?
Esequential_23/sequential_22/conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Gsequential_23/sequential_22/conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Gsequential_23/sequential_22/conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential_23/sequential_22/conv2d_transpose_15/strided_slice_1StridedSlice>sequential_23/sequential_22/conv2d_transpose_15/stack:output:0Nsequential_23/sequential_22/conv2d_transpose_15/strided_slice_1/stack:output:0Psequential_23/sequential_22/conv2d_transpose_15/strided_slice_1/stack_1:output:0Psequential_23/sequential_22/conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Osequential_23/sequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOpXsequential_23_sequential_22_conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
@sequential_23/sequential_22/conv2d_transpose_15/conv2d_transposeConv2DBackpropInput>sequential_23/sequential_22/conv2d_transpose_15/stack:output:0Wsequential_23/sequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0Gsequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
Fsequential_23/sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOpOsequential_23_sequential_22_conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
7sequential_23/sequential_22/conv2d_transpose_15/BiasAddBiasAddIsequential_23/sequential_22/conv2d_transpose_15/conv2d_transpose:output:0Nsequential_23/sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
.sequential_23/sequential_22/activation_31/TanhTanh@sequential_23/sequential_22/conv2d_transpose_15/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
;sequential_23/sequential_21/conv2d_14/Conv2D/ReadVariableOpReadVariableOpDsequential_23_sequential_21_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
,sequential_23/sequential_21/conv2d_14/Conv2DConv2D2sequential_23/sequential_22/activation_31/Tanh:y:0Csequential_23/sequential_21/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
<sequential_23/sequential_21/conv2d_14/BiasAdd/ReadVariableOpReadVariableOpEsequential_23_sequential_21_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
-sequential_23/sequential_21/conv2d_14/BiasAddBiasAdd5sequential_23/sequential_21/conv2d_14/Conv2D:output:0Dsequential_23/sequential_21/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
4sequential_23/sequential_21/leaky_re_lu_21/LeakyRelu	LeakyRelu6sequential_23/sequential_21/conv2d_14/BiasAdd:output:0*/
_output_shapes
:????????? ?
;sequential_23/sequential_21/conv2d_15/Conv2D/ReadVariableOpReadVariableOpDsequential_23_sequential_21_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
,sequential_23/sequential_21/conv2d_15/Conv2DConv2DBsequential_23/sequential_21/leaky_re_lu_21/LeakyRelu:activations:0Csequential_23/sequential_21/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
<sequential_23/sequential_21/conv2d_15/BiasAdd/ReadVariableOpReadVariableOpEsequential_23_sequential_21_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
-sequential_23/sequential_21/conv2d_15/BiasAddBiasAdd5sequential_23/sequential_21/conv2d_15/Conv2D:output:0Dsequential_23/sequential_21/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
4sequential_23/sequential_21/leaky_re_lu_22/LeakyRelu	LeakyRelu6sequential_23/sequential_21/conv2d_15/BiasAdd:output:0*/
_output_shapes
:?????????@|
+sequential_23/sequential_21/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
-sequential_23/sequential_21/flatten_7/ReshapeReshapeBsequential_23/sequential_21/leaky_re_lu_22/LeakyRelu:activations:04sequential_23/sequential_21/flatten_7/Const:output:0*
T0*(
_output_shapes
:???????????
:sequential_23/sequential_21/dense_28/MatMul/ReadVariableOpReadVariableOpCsequential_23_sequential_21_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
+sequential_23/sequential_21/dense_28/MatMulMatMul6sequential_23/sequential_21/flatten_7/Reshape:output:0Bsequential_23/sequential_21/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
;sequential_23/sequential_21/dense_28/BiasAdd/ReadVariableOpReadVariableOpDsequential_23_sequential_21_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,sequential_23/sequential_21/dense_28/BiasAddBiasAdd5sequential_23/sequential_21/dense_28/MatMul:product:0Csequential_23/sequential_21/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
4sequential_23/sequential_21/leaky_re_lu_23/LeakyRelu	LeakyRelu5sequential_23/sequential_21/dense_28/BiasAdd:output:0*(
_output_shapes
:???????????
:sequential_23/sequential_21/dense_29/MatMul/ReadVariableOpReadVariableOpCsequential_23_sequential_21_dense_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
+sequential_23/sequential_21/dense_29/MatMulMatMulBsequential_23/sequential_21/leaky_re_lu_23/LeakyRelu:activations:0Bsequential_23/sequential_21/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;sequential_23/sequential_21/dense_29/BiasAdd/ReadVariableOpReadVariableOpDsequential_23_sequential_21_dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
,sequential_23/sequential_21/dense_29/BiasAddBiasAdd5sequential_23/sequential_21/dense_29/MatMul:product:0Csequential_23/sequential_21/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,sequential_23/sequential_21/dense_29/SigmoidSigmoid5sequential_23/sequential_21/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
IdentityIdentity0sequential_23/sequential_21/dense_29/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp=^sequential_23/sequential_21/conv2d_14/BiasAdd/ReadVariableOp<^sequential_23/sequential_21/conv2d_14/Conv2D/ReadVariableOp=^sequential_23/sequential_21/conv2d_15/BiasAdd/ReadVariableOp<^sequential_23/sequential_21/conv2d_15/Conv2D/ReadVariableOp<^sequential_23/sequential_21/dense_28/BiasAdd/ReadVariableOp;^sequential_23/sequential_21/dense_28/MatMul/ReadVariableOp<^sequential_23/sequential_21/dense_29/BiasAdd/ReadVariableOp;^sequential_23/sequential_21/dense_29/MatMul/ReadVariableOpL^sequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOpN^sequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_1N^sequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_2P^sequential_23/sequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOpL^sequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOpN^sequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_1N^sequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_2P^sequential_23/sequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOpS^sequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOpU^sequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1B^sequential_23/sequential_22/batch_normalization_23/ReadVariableOpD^sequential_23/sequential_22/batch_normalization_23/ReadVariableOp_1G^sequential_23/sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOpP^sequential_23/sequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOpG^sequential_23/sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOpP^sequential_23/sequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOp<^sequential_23/sequential_22/dense_30/BiasAdd/ReadVariableOp;^sequential_23/sequential_22/dense_30/MatMul/ReadVariableOp<^sequential_23/sequential_22/dense_31/BiasAdd/ReadVariableOp;^sequential_23/sequential_22/dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2|
<sequential_23/sequential_21/conv2d_14/BiasAdd/ReadVariableOp<sequential_23/sequential_21/conv2d_14/BiasAdd/ReadVariableOp2z
;sequential_23/sequential_21/conv2d_14/Conv2D/ReadVariableOp;sequential_23/sequential_21/conv2d_14/Conv2D/ReadVariableOp2|
<sequential_23/sequential_21/conv2d_15/BiasAdd/ReadVariableOp<sequential_23/sequential_21/conv2d_15/BiasAdd/ReadVariableOp2z
;sequential_23/sequential_21/conv2d_15/Conv2D/ReadVariableOp;sequential_23/sequential_21/conv2d_15/Conv2D/ReadVariableOp2z
;sequential_23/sequential_21/dense_28/BiasAdd/ReadVariableOp;sequential_23/sequential_21/dense_28/BiasAdd/ReadVariableOp2x
:sequential_23/sequential_21/dense_28/MatMul/ReadVariableOp:sequential_23/sequential_21/dense_28/MatMul/ReadVariableOp2z
;sequential_23/sequential_21/dense_29/BiasAdd/ReadVariableOp;sequential_23/sequential_21/dense_29/BiasAdd/ReadVariableOp2x
:sequential_23/sequential_21/dense_29/MatMul/ReadVariableOp:sequential_23/sequential_21/dense_29/MatMul/ReadVariableOp2?
Ksequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOpKsequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp2?
Msequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_1Msequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_12?
Msequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_2Msequential_23/sequential_22/batch_normalization_21/batchnorm/ReadVariableOp_22?
Osequential_23/sequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOpOsequential_23/sequential_22/batch_normalization_21/batchnorm/mul/ReadVariableOp2?
Ksequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOpKsequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp2?
Msequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_1Msequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_12?
Msequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_2Msequential_23/sequential_22/batch_normalization_22/batchnorm/ReadVariableOp_22?
Osequential_23/sequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOpOsequential_23/sequential_22/batch_normalization_22/batchnorm/mul/ReadVariableOp2?
Rsequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOpRsequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2?
Tsequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Tsequential_23/sequential_22/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12?
Asequential_23/sequential_22/batch_normalization_23/ReadVariableOpAsequential_23/sequential_22/batch_normalization_23/ReadVariableOp2?
Csequential_23/sequential_22/batch_normalization_23/ReadVariableOp_1Csequential_23/sequential_22/batch_normalization_23/ReadVariableOp_12?
Fsequential_23/sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOpFsequential_23/sequential_22/conv2d_transpose_14/BiasAdd/ReadVariableOp2?
Osequential_23/sequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOpOsequential_23/sequential_22/conv2d_transpose_14/conv2d_transpose/ReadVariableOp2?
Fsequential_23/sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOpFsequential_23/sequential_22/conv2d_transpose_15/BiasAdd/ReadVariableOp2?
Osequential_23/sequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOpOsequential_23/sequential_22/conv2d_transpose_15/conv2d_transpose/ReadVariableOp2z
;sequential_23/sequential_22/dense_30/BiasAdd/ReadVariableOp;sequential_23/sequential_22/dense_30/BiasAdd/ReadVariableOp2x
:sequential_23/sequential_22/dense_30/MatMul/ReadVariableOp:sequential_23/sequential_22/dense_30/MatMul/ReadVariableOp2z
;sequential_23/sequential_22/dense_31/BiasAdd/ReadVariableOp;sequential_23/sequential_22/dense_31/BiasAdd/ReadVariableOp2x
:sequential_23/sequential_22/dense_31/MatMul/ReadVariableOp:sequential_23/sequential_22/dense_31/MatMul/ReadVariableOp:\ X
'
_output_shapes
:?????????d
-
_user_specified_namesequential_22_input
?=
?

J__inference_sequential_22_layer_call_and_return_conditional_losses_1083664

inputs#
dense_30_1083561:	d?
dense_30_1083563:	?-
batch_normalization_21_1083573:	?-
batch_normalization_21_1083575:	?-
batch_normalization_21_1083577:	?-
batch_normalization_21_1083579:	?$
dense_31_1083593:
??
dense_31_1083595:	?-
batch_normalization_22_1083605:	?-
batch_normalization_22_1083607:	?-
batch_normalization_22_1083609:	?-
batch_normalization_22_1083611:	?5
conv2d_transpose_14_1083630: @)
conv2d_transpose_14_1083632: ,
batch_normalization_23_1083642: ,
batch_normalization_23_1083644: ,
batch_normalization_23_1083646: ,
batch_normalization_23_1083648: 5
conv2d_transpose_15_1083651: )
conv2d_transpose_15_1083653:
identity??.batch_normalization_21/StatefulPartitionedCall?.batch_normalization_22/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCallinputsdense_30_1083561dense_30_1083563*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_30_layer_call_and_return_conditional_losses_1083560?
activation_28/PartitionedCallPartitionedCall)dense_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_28_layer_call_and_return_conditional_losses_1083571?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall&activation_28/PartitionedCall:output:0batch_normalization_21_1083573batch_normalization_21_1083575batch_normalization_21_1083577batch_normalization_21_1083579*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1083251?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0dense_31_1083593dense_31_1083595*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_1083592?
activation_29/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_29_layer_call_and_return_conditional_losses_1083603?
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall&activation_29/PartitionedCall:output:0batch_normalization_22_1083605batch_normalization_22_1083607batch_normalization_22_1083609batch_normalization_22_1083611*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1083333?
reshape_7/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_reshape_7_layer_call_and_return_conditional_losses_1083628?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv2d_transpose_14_1083630conv2d_transpose_14_1083632*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_1083428?
activation_30/PartitionedCallPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_30_layer_call_and_return_conditional_losses_1083640?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall&activation_30/PartitionedCall:output:0batch_normalization_23_1083642batch_normalization_23_1083644batch_normalization_23_1083646batch_normalization_23_1083648*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1083457?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0conv2d_transpose_15_1083651conv2d_transpose_15_1083653*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_1083536?
activation_31/PartitionedCallPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_31_layer_call_and_return_conditional_losses_1083661}
IdentityIdentity&activation_31/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?9
#__inference__traced_restore_1086937
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: 5
"assignvariableop_5_dense_30_kernel:	d?/
 assignvariableop_6_dense_30_bias:	?>
/assignvariableop_7_batch_normalization_21_gamma:	?=
.assignvariableop_8_batch_normalization_21_beta:	?D
5assignvariableop_9_batch_normalization_21_moving_mean:	?I
:assignvariableop_10_batch_normalization_21_moving_variance:	?7
#assignvariableop_11_dense_31_kernel:
??0
!assignvariableop_12_dense_31_bias:	??
0assignvariableop_13_batch_normalization_22_gamma:	?>
/assignvariableop_14_batch_normalization_22_beta:	?E
6assignvariableop_15_batch_normalization_22_moving_mean:	?I
:assignvariableop_16_batch_normalization_22_moving_variance:	?H
.assignvariableop_17_conv2d_transpose_14_kernel: @:
,assignvariableop_18_conv2d_transpose_14_bias: >
0assignvariableop_19_batch_normalization_23_gamma: =
/assignvariableop_20_batch_normalization_23_beta: D
6assignvariableop_21_batch_normalization_23_moving_mean: H
:assignvariableop_22_batch_normalization_23_moving_variance: H
.assignvariableop_23_conv2d_transpose_15_kernel: :
,assignvariableop_24_conv2d_transpose_15_bias:>
$assignvariableop_25_conv2d_14_kernel: 0
"assignvariableop_26_conv2d_14_bias: >
$assignvariableop_27_conv2d_15_kernel: @0
"assignvariableop_28_conv2d_15_bias:@7
#assignvariableop_29_dense_28_kernel:
??0
!assignvariableop_30_dense_28_bias:	?6
#assignvariableop_31_dense_29_kernel:	?/
!assignvariableop_32_dense_29_bias:)
assignvariableop_33_adam_iter_1:	 +
!assignvariableop_34_adam_beta_1_1: +
!assignvariableop_35_adam_beta_2_1: *
 assignvariableop_36_adam_decay_1: 2
(assignvariableop_37_adam_learning_rate_1: #
assignvariableop_38_total: #
assignvariableop_39_count: %
assignvariableop_40_total_1: %
assignvariableop_41_count_1: %
assignvariableop_42_total_2: %
assignvariableop_43_count_2: =
*assignvariableop_44_adam_dense_30_kernel_m:	d?7
(assignvariableop_45_adam_dense_30_bias_m:	?F
7assignvariableop_46_adam_batch_normalization_21_gamma_m:	?E
6assignvariableop_47_adam_batch_normalization_21_beta_m:	?>
*assignvariableop_48_adam_dense_31_kernel_m:
??7
(assignvariableop_49_adam_dense_31_bias_m:	?F
7assignvariableop_50_adam_batch_normalization_22_gamma_m:	?E
6assignvariableop_51_adam_batch_normalization_22_beta_m:	?O
5assignvariableop_52_adam_conv2d_transpose_14_kernel_m: @A
3assignvariableop_53_adam_conv2d_transpose_14_bias_m: E
7assignvariableop_54_adam_batch_normalization_23_gamma_m: D
6assignvariableop_55_adam_batch_normalization_23_beta_m: O
5assignvariableop_56_adam_conv2d_transpose_15_kernel_m: A
3assignvariableop_57_adam_conv2d_transpose_15_bias_m:=
*assignvariableop_58_adam_dense_30_kernel_v:	d?7
(assignvariableop_59_adam_dense_30_bias_v:	?F
7assignvariableop_60_adam_batch_normalization_21_gamma_v:	?E
6assignvariableop_61_adam_batch_normalization_21_beta_v:	?>
*assignvariableop_62_adam_dense_31_kernel_v:
??7
(assignvariableop_63_adam_dense_31_bias_v:	?F
7assignvariableop_64_adam_batch_normalization_22_gamma_v:	?E
6assignvariableop_65_adam_batch_normalization_22_beta_v:	?O
5assignvariableop_66_adam_conv2d_transpose_14_kernel_v: @A
3assignvariableop_67_adam_conv2d_transpose_14_bias_v: E
7assignvariableop_68_adam_batch_normalization_23_gamma_v: D
6assignvariableop_69_adam_batch_normalization_23_beta_v: O
5assignvariableop_70_adam_conv2d_transpose_15_kernel_v: A
3assignvariableop_71_adam_conv2d_transpose_15_bias_v:E
+assignvariableop_72_adam_conv2d_14_kernel_m: 7
)assignvariableop_73_adam_conv2d_14_bias_m: E
+assignvariableop_74_adam_conv2d_15_kernel_m: @7
)assignvariableop_75_adam_conv2d_15_bias_m:@>
*assignvariableop_76_adam_dense_28_kernel_m:
??7
(assignvariableop_77_adam_dense_28_bias_m:	?=
*assignvariableop_78_adam_dense_29_kernel_m:	?6
(assignvariableop_79_adam_dense_29_bias_m:E
+assignvariableop_80_adam_conv2d_14_kernel_v: 7
)assignvariableop_81_adam_conv2d_14_bias_v: E
+assignvariableop_82_adam_conv2d_15_kernel_v: @7
)assignvariableop_83_adam_conv2d_15_bias_v:@>
*assignvariableop_84_adam_dense_28_kernel_v:
??7
(assignvariableop_85_adam_dense_28_bias_v:	?=
*assignvariableop_86_adam_dense_29_kernel_v:	?6
(assignvariableop_87_adam_dense_29_bias_v:
identity_89??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_9?+
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*?+
value?+B?*YB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/20/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/21/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/22/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/23/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/24/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/25/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/26/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/27/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/20/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/21/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/22/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/23/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/24/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/25/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/26/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/27/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Y*
dtype0*?
value?B?YB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*g
dtypes]
[2Y		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_30_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_30_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_21_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_21_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp5assignvariableop_9_batch_normalization_21_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp:assignvariableop_10_batch_normalization_21_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_31_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_31_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp0assignvariableop_13_batch_normalization_22_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_22_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp6assignvariableop_15_batch_normalization_22_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp:assignvariableop_16_batch_normalization_22_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp.assignvariableop_17_conv2d_transpose_14_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp,assignvariableop_18_conv2d_transpose_14_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp0assignvariableop_19_batch_normalization_23_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_23_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp6assignvariableop_21_batch_normalization_23_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp:assignvariableop_22_batch_normalization_23_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp.assignvariableop_23_conv2d_transpose_15_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp,assignvariableop_24_conv2d_transpose_15_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp$assignvariableop_25_conv2d_14_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp"assignvariableop_26_conv2d_14_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp$assignvariableop_27_conv2d_15_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp"assignvariableop_28_conv2d_15_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp#assignvariableop_29_dense_28_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp!assignvariableop_30_dense_28_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp#assignvariableop_31_dense_29_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp!assignvariableop_32_dense_29_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_iter_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp!assignvariableop_34_adam_beta_1_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp!assignvariableop_35_adam_beta_2_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp assignvariableop_36_adam_decay_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_learning_rate_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpassignvariableop_38_totalIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpassignvariableop_39_countIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpassignvariableop_40_total_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpassignvariableop_41_count_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOpassignvariableop_42_total_2Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOpassignvariableop_43_count_2Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_dense_30_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_dense_30_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp7assignvariableop_46_adam_batch_normalization_21_gamma_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_batch_normalization_21_beta_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_dense_31_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_dense_31_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp7assignvariableop_50_adam_batch_normalization_22_gamma_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_batch_normalization_22_beta_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp5assignvariableop_52_adam_conv2d_transpose_14_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp3assignvariableop_53_adam_conv2d_transpose_14_bias_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp7assignvariableop_54_adam_batch_normalization_23_gamma_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_batch_normalization_23_beta_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp5assignvariableop_56_adam_conv2d_transpose_15_kernel_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp3assignvariableop_57_adam_conv2d_transpose_15_bias_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_dense_30_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_dense_30_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp7assignvariableop_60_adam_batch_normalization_21_gamma_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_21_beta_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_dense_31_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_dense_31_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp7assignvariableop_64_adam_batch_normalization_22_gamma_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_22_beta_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_conv2d_transpose_14_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp3assignvariableop_67_adam_conv2d_transpose_14_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_batch_normalization_23_gamma_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_23_beta_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_conv2d_transpose_15_kernel_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp3assignvariableop_71_adam_conv2d_transpose_15_bias_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp+assignvariableop_72_adam_conv2d_14_kernel_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp)assignvariableop_73_adam_conv2d_14_bias_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp+assignvariableop_74_adam_conv2d_15_kernel_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp)assignvariableop_75_adam_conv2d_15_bias_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp*assignvariableop_76_adam_dense_28_kernel_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp(assignvariableop_77_adam_dense_28_bias_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp*assignvariableop_78_adam_dense_29_kernel_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp(assignvariableop_79_adam_dense_29_bias_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp+assignvariableop_80_adam_conv2d_14_kernel_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp)assignvariableop_81_adam_conv2d_14_bias_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp+assignvariableop_82_adam_conv2d_15_kernel_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp)assignvariableop_83_adam_conv2d_15_bias_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp*assignvariableop_84_adam_dense_28_kernel_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp(assignvariableop_85_adam_dense_28_bias_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp*assignvariableop_86_adam_dense_29_kernel_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp(assignvariableop_87_adam_dense_29_bias_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_88Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_89IdentityIdentity_88:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_89Identity_89:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
/__inference_sequential_21_layer_call_fn_1085766

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084162o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_30_layer_call_fn_1085864

inputs
unknown:	d?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_30_layer_call_and_return_conditional_losses_1083560p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
? 
?
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_1086248

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?=
?

J__inference_sequential_22_layer_call_and_return_conditional_losses_1084060
dense_30_input#
dense_30_1084007:	d?
dense_30_1084009:	?-
batch_normalization_21_1084013:	?-
batch_normalization_21_1084015:	?-
batch_normalization_21_1084017:	?-
batch_normalization_21_1084019:	?$
dense_31_1084022:
??
dense_31_1084024:	?-
batch_normalization_22_1084028:	?-
batch_normalization_22_1084030:	?-
batch_normalization_22_1084032:	?-
batch_normalization_22_1084034:	?5
conv2d_transpose_14_1084038: @)
conv2d_transpose_14_1084040: ,
batch_normalization_23_1084044: ,
batch_normalization_23_1084046: ,
batch_normalization_23_1084048: ,
batch_normalization_23_1084050: 5
conv2d_transpose_15_1084053: )
conv2d_transpose_15_1084055:
identity??.batch_normalization_21/StatefulPartitionedCall?.batch_normalization_22/StatefulPartitionedCall?.batch_normalization_23/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCalldense_30_inputdense_30_1084007dense_30_1084009*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_30_layer_call_and_return_conditional_losses_1083560?
activation_28/PartitionedCallPartitionedCall)dense_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_28_layer_call_and_return_conditional_losses_1083571?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall&activation_28/PartitionedCall:output:0batch_normalization_21_1084013batch_normalization_21_1084015batch_normalization_21_1084017batch_normalization_21_1084019*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1083298?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0dense_31_1084022dense_31_1084024*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_1083592?
activation_29/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_29_layer_call_and_return_conditional_losses_1083603?
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall&activation_29/PartitionedCall:output:0batch_normalization_22_1084028batch_normalization_22_1084030batch_normalization_22_1084032batch_normalization_22_1084034*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1083380?
reshape_7/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_reshape_7_layer_call_and_return_conditional_losses_1083628?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0conv2d_transpose_14_1084038conv2d_transpose_14_1084040*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_1083428?
activation_30/PartitionedCallPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_30_layer_call_and_return_conditional_losses_1083640?
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall&activation_30/PartitionedCall:output:0batch_normalization_23_1084044batch_normalization_23_1084046batch_normalization_23_1084048batch_normalization_23_1084050*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1083488?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0conv2d_transpose_15_1084053conv2d_transpose_15_1084055*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_1083536?
activation_31/PartitionedCallPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_activation_31_layer_call_and_return_conditional_losses_1083661}
IdentityIdentity&activation_31/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:W S
'
_output_shapes
:?????????d
(
_user_specified_namedense_30_input
?

?
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1084100

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?(
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1085821

inputsB
(conv2d_14_conv2d_readvariableop_resource: 7
)conv2d_14_biasadd_readvariableop_resource: B
(conv2d_15_conv2d_readvariableop_resource: @7
)conv2d_15_biasadd_readvariableop_resource:@;
'dense_28_matmul_readvariableop_resource:
??7
(dense_28_biasadd_readvariableop_resource:	?:
'dense_29_matmul_readvariableop_resource:	?6
(dense_29_biasadd_readvariableop_resource:
identity?? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?dense_28/MatMul/ReadVariableOp?dense_29/BiasAdd/ReadVariableOp?dense_29/MatMul/ReadVariableOp?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_14/Conv2DConv2Dinputs'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? r
leaky_re_lu_21/LeakyRelu	LeakyReluconv2d_14/BiasAdd:output:0*/
_output_shapes
:????????? ?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_15/Conv2DConv2D&leaky_re_lu_21/LeakyRelu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@r
leaky_re_lu_22/LeakyRelu	LeakyReluconv2d_15/BiasAdd:output:0*/
_output_shapes
:?????????@`
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
flatten_7/ReshapeReshape&leaky_re_lu_22/LeakyRelu:activations:0flatten_7/Const:output:0*
T0*(
_output_shapes
:???????????
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_28/MatMulMatMulflatten_7/Reshape:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????j
leaky_re_lu_23/LeakyRelu	LeakyReludense_28/BiasAdd:output:0*(
_output_shapes
:???????????
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_29/MatMulMatMul&leaky_re_lu_23/LeakyRelu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_29/SigmoidSigmoiddense_29/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_29/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
L
0__inference_leaky_re_lu_22_layer_call_fn_1086311

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1084111h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
G
+__inference_reshape_7_layer_call_fn_1086078

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_reshape_7_layer_call_and_return_conditional_losses_1083628h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1086206

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
F__inference_conv2d_14_layer_call_and_return_conditional_losses_1084077

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?

J__inference_sequential_23_layer_call_and_return_conditional_losses_1084642

inputs(
sequential_22_1084583:	d?$
sequential_22_1084585:	?$
sequential_22_1084587:	?$
sequential_22_1084589:	?$
sequential_22_1084591:	?$
sequential_22_1084593:	?)
sequential_22_1084595:
??$
sequential_22_1084597:	?$
sequential_22_1084599:	?$
sequential_22_1084601:	?$
sequential_22_1084603:	?$
sequential_22_1084605:	?/
sequential_22_1084607: @#
sequential_22_1084609: #
sequential_22_1084611: #
sequential_22_1084613: #
sequential_22_1084615: #
sequential_22_1084617: /
sequential_22_1084619: #
sequential_22_1084621:/
sequential_21_1084624: #
sequential_21_1084626: /
sequential_21_1084628: @#
sequential_21_1084630:@)
sequential_21_1084632:
??$
sequential_21_1084634:	?(
sequential_21_1084636:	?#
sequential_21_1084638:
identity??%sequential_21/StatefulPartitionedCall?%sequential_22/StatefulPartitionedCall?
%sequential_22/StatefulPartitionedCallStatefulPartitionedCallinputssequential_22_1084583sequential_22_1084585sequential_22_1084587sequential_22_1084589sequential_22_1084591sequential_22_1084593sequential_22_1084595sequential_22_1084597sequential_22_1084599sequential_22_1084601sequential_22_1084603sequential_22_1084605sequential_22_1084607sequential_22_1084609sequential_22_1084611sequential_22_1084613sequential_22_1084615sequential_22_1084617sequential_22_1084619sequential_22_1084621* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*0
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_22_layer_call_and_return_conditional_losses_1083860?
%sequential_21/StatefulPartitionedCallStatefulPartitionedCall.sequential_22/StatefulPartitionedCall:output:0sequential_21_1084624sequential_21_1084626sequential_21_1084628sequential_21_1084630sequential_21_1084632sequential_21_1084634sequential_21_1084636sequential_21_1084638*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084296}
IdentityIdentity.sequential_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^sequential_21/StatefulPartitionedCall&^sequential_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%sequential_21/StatefulPartitionedCall%sequential_21/StatefulPartitionedCall2N
%sequential_22/StatefulPartitionedCall%sequential_22/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1086306

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
b
F__inference_reshape_7_layer_call_and_return_conditional_losses_1083628

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_22_layer_call_fn_1086006

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1083333p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_activation_28_layer_call_and_return_conditional_losses_1083571

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:??????????[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
sequential_22_input<
%serving_default_sequential_22_input:0?????????dA
sequential_210
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures"
_tf_keras_sequential
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer-8
layer_with_weights-5
layer-9
layer_with_weights-6
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
layer_with_weights-0
layer-0
layer-1
 layer_with_weights-1
 layer-2
!layer-3
"layer-4
#layer_with_weights-2
#layer-5
$layer-6
%layer_with_weights-3
%layer-7
&	optimizer
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
-iter

.beta_1

/beta_2
	0decay
1learning_rate2m?3m?4m?5m?8m?9m?:m?;m?>m??m?@m?Am?Dm?Em?2v?3v?4v?5v?8v?9v?:v?;v?>v??v?@v?Av?Dv?Ev?"
	optimizer
?
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
B16
C17
D18
E19
F20
G21
H22
I23
J24
K25
L26
M27"
trackable_list_wrapper
?
20
31
42
53
84
95
:6
;7
>8
?9
@10
A11
D12
E13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_sequential_23_layer_call_fn_1084517
/__inference_sequential_23_layer_call_fn_1084953
/__inference_sequential_23_layer_call_fn_1085014
/__inference_sequential_23_layer_call_fn_1084762?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_23_layer_call_and_return_conditional_losses_1085158
J__inference_sequential_23_layer_call_and_return_conditional_losses_1085330
J__inference_sequential_23_layer_call_and_return_conditional_losses_1084824
J__inference_sequential_23_layer_call_and_return_conditional_losses_1084886?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_1083227sequential_22_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
Sserving_default"
signature_map
?

2kernel
3bias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses"
_tf_keras_layer
?
`axis
	4gamma
5beta
6moving_mean
7moving_variance
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
?

8kernel
9bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
?
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
?
saxis
	:gamma
;beta
<moving_mean
=moving_variance
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
?
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

>kernel
?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Dkernel
Ebias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
B16
C17
D18
E19"
trackable_list_wrapper
?
20
31
42
53
84
95
:6
;7
>8
?9
@10
A11
D12
E13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_sequential_22_layer_call_fn_1083707
/__inference_sequential_22_layer_call_fn_1085438
/__inference_sequential_22_layer_call_fn_1085483
/__inference_sequential_22_layer_call_fn_1083948?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_22_layer_call_and_return_conditional_losses_1085597
J__inference_sequential_22_layer_call_and_return_conditional_losses_1085739
J__inference_sequential_22_layer_call_and_return_conditional_losses_1084004
J__inference_sequential_22_layer_call_and_return_conditional_losses_1084060?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?

Fkernel
Gbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Hkernel
Ibias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Jkernel
Kbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Lkernel
Mbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateFm?Gm?Hm?Im?Jm?Km?Lm?Mm?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?"
	optimizer
X
F0
G1
H2
I3
J4
K5
L6
M7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_sequential_21_layer_call_fn_1084181
/__inference_sequential_21_layer_call_fn_1085766
/__inference_sequential_21_layer_call_fn_1085787
/__inference_sequential_21_layer_call_fn_1084336?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1085821
J__inference_sequential_21_layer_call_and_return_conditional_losses_1085855
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084364
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084392?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
": 	d?2dense_30/kernel
:?2dense_30/bias
+:)?2batch_normalization_21/gamma
*:(?2batch_normalization_21/beta
3:1? (2"batch_normalization_21/moving_mean
7:5? (2&batch_normalization_21/moving_variance
#:!
??2dense_31/kernel
:?2dense_31/bias
+:)?2batch_normalization_22/gamma
*:(?2batch_normalization_22/beta
3:1? (2"batch_normalization_22/moving_mean
7:5? (2&batch_normalization_22/moving_variance
4:2 @2conv2d_transpose_14/kernel
&:$ 2conv2d_transpose_14/bias
*:( 2batch_normalization_23/gamma
):' 2batch_normalization_23/beta
2:0  (2"batch_normalization_23/moving_mean
6:4  (2&batch_normalization_23/moving_variance
4:2 2conv2d_transpose_15/kernel
&:$2conv2d_transpose_15/bias
*:( 2conv2d_14/kernel
: 2conv2d_14/bias
*:( @2conv2d_15/kernel
:@2conv2d_15/bias
#:!
??2dense_28/kernel
:?2dense_28/bias
": 	?2dense_29/kernel
:2dense_29/bias
?
60
71
<2
=3
B4
C5
F6
G7
H8
I9
J10
K11
L12
M13"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_signature_wrapper_1085393sequential_22_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_30_layer_call_fn_1085864?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_30_layer_call_and_return_conditional_losses_1085874?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_activation_28_layer_call_fn_1085879?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_activation_28_layer_call_and_return_conditional_losses_1085884?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
40
51
62
73"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_21_layer_call_fn_1085897
8__inference_batch_normalization_21_layer_call_fn_1085910?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1085930
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1085964?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_31_layer_call_fn_1085973?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_31_layer_call_and_return_conditional_losses_1085983?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_activation_29_layer_call_fn_1085988?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_activation_29_layer_call_and_return_conditional_losses_1085993?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
:0
;1
<2
=3"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_22_layer_call_fn_1086006
8__inference_batch_normalization_22_layer_call_fn_1086019?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1086039
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1086073?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_reshape_7_layer_call_fn_1086078?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_reshape_7_layer_call_and_return_conditional_losses_1086092?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
5__inference_conv2d_transpose_14_layer_call_fn_1086101?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_1086134?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_activation_30_layer_call_fn_1086139?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_activation_30_layer_call_and_return_conditional_losses_1086144?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
@0
A1
B2
C3"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_23_layer_call_fn_1086157
8__inference_batch_normalization_23_layer_call_fn_1086170?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1086188
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1086206?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
5__inference_conv2d_transpose_15_layer_call_fn_1086215?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_1086248?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_activation_31_layer_call_fn_1086253?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_activation_31_layer_call_and_return_conditional_losses_1086258?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
J
60
71
<2
=3
B4
C5"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_conv2d_14_layer_call_fn_1086267?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_14_layer_call_and_return_conditional_losses_1086277?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_leaky_re_lu_21_layer_call_fn_1086282?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1086287?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_conv2d_15_layer_call_fn_1086296?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1086306?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_leaky_re_lu_22_layer_call_fn_1086311?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1086316?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_flatten_7_layer_call_fn_1086321?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_flatten_7_layer_call_and_return_conditional_losses_1086327?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_28_layer_call_fn_1086336?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_28_layer_call_and_return_conditional_losses_1086346?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_leaky_re_lu_23_layer_call_fn_1086351?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1086356?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_29_layer_call_fn_1086365?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_29_layer_call_and_return_conditional_losses_1086376?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
X
F0
G1
H2
I3
J4
K5
L6
M7"
trackable_list_wrapper
X
0
1
 2
!3
"4
#5
$6
%7"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
':%	d?2Adam/dense_30/kernel/m
!:?2Adam/dense_30/bias/m
0:.?2#Adam/batch_normalization_21/gamma/m
/:-?2"Adam/batch_normalization_21/beta/m
(:&
??2Adam/dense_31/kernel/m
!:?2Adam/dense_31/bias/m
0:.?2#Adam/batch_normalization_22/gamma/m
/:-?2"Adam/batch_normalization_22/beta/m
9:7 @2!Adam/conv2d_transpose_14/kernel/m
+:) 2Adam/conv2d_transpose_14/bias/m
/:- 2#Adam/batch_normalization_23/gamma/m
.:, 2"Adam/batch_normalization_23/beta/m
9:7 2!Adam/conv2d_transpose_15/kernel/m
+:)2Adam/conv2d_transpose_15/bias/m
':%	d?2Adam/dense_30/kernel/v
!:?2Adam/dense_30/bias/v
0:.?2#Adam/batch_normalization_21/gamma/v
/:-?2"Adam/batch_normalization_21/beta/v
(:&
??2Adam/dense_31/kernel/v
!:?2Adam/dense_31/bias/v
0:.?2#Adam/batch_normalization_22/gamma/v
/:-?2"Adam/batch_normalization_22/beta/v
9:7 @2!Adam/conv2d_transpose_14/kernel/v
+:) 2Adam/conv2d_transpose_14/bias/v
/:- 2#Adam/batch_normalization_23/gamma/v
.:, 2"Adam/batch_normalization_23/beta/v
9:7 2!Adam/conv2d_transpose_15/kernel/v
+:)2Adam/conv2d_transpose_15/bias/v
/:- 2Adam/conv2d_14/kernel/m
!: 2Adam/conv2d_14/bias/m
/:- @2Adam/conv2d_15/kernel/m
!:@2Adam/conv2d_15/bias/m
(:&
??2Adam/dense_28/kernel/m
!:?2Adam/dense_28/bias/m
':%	?2Adam/dense_29/kernel/m
 :2Adam/dense_29/bias/m
/:- 2Adam/conv2d_14/kernel/v
!: 2Adam/conv2d_14/bias/v
/:- @2Adam/conv2d_15/kernel/v
!:@2Adam/conv2d_15/bias/v
(:&
??2Adam/dense_28/kernel/v
!:?2Adam/dense_28/bias/v
':%	?2Adam/dense_29/kernel/v
 :2Adam/dense_29/bias/v?
"__inference__wrapped_model_1083227?23746589=:<;>?@ABCDEFGHIJKLM<?9
2?/
-?*
sequential_22_input?????????d
? "=?:
8
sequential_21'?$
sequential_21??????????
J__inference_activation_28_layer_call_and_return_conditional_losses_1085884Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
/__inference_activation_28_layer_call_fn_1085879M0?-
&?#
!?
inputs??????????
? "????????????
J__inference_activation_29_layer_call_and_return_conditional_losses_1085993Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
/__inference_activation_29_layer_call_fn_1085988M0?-
&?#
!?
inputs??????????
? "????????????
J__inference_activation_30_layer_call_and_return_conditional_losses_1086144h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
/__inference_activation_30_layer_call_fn_1086139[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
J__inference_activation_31_layer_call_and_return_conditional_losses_1086258h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
/__inference_activation_31_layer_call_fn_1086253[7?4
-?*
(?%
inputs?????????
? " ???????????
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1085930d74654?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1085964d67454?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
8__inference_batch_normalization_21_layer_call_fn_1085897W74654?1
*?'
!?
inputs??????????
p 
? "????????????
8__inference_batch_normalization_21_layer_call_fn_1085910W67454?1
*?'
!?
inputs??????????
p
? "????????????
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1086039d=:<;4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1086073d<=:;4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
8__inference_batch_normalization_22_layer_call_fn_1086006W=:<;4?1
*?'
!?
inputs??????????
p 
? "????????????
8__inference_batch_normalization_22_layer_call_fn_1086019W<=:;4?1
*?'
!?
inputs??????????
p
? "????????????
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1086188?@ABCM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1086206?@ABCM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
8__inference_batch_normalization_23_layer_call_fn_1086157?@ABCM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_23_layer_call_fn_1086170?@ABCM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
F__inference_conv2d_14_layer_call_and_return_conditional_losses_1086277lFG7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
+__inference_conv2d_14_layer_call_fn_1086267_FG7?4
-?*
(?%
inputs?????????
? " ?????????? ?
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1086306lHI7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
+__inference_conv2d_15_layer_call_fn_1086296_HI7?4
-?*
(?%
inputs????????? 
? " ??????????@?
P__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_1086134?>?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
5__inference_conv2d_transpose_14_layer_call_fn_1086101?>?I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
P__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_1086248?DEI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
5__inference_conv2d_transpose_15_layer_call_fn_1086215?DEI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
E__inference_dense_28_layer_call_and_return_conditional_losses_1086346^JK0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_28_layer_call_fn_1086336QJK0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_29_layer_call_and_return_conditional_losses_1086376]LM0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ~
*__inference_dense_29_layer_call_fn_1086365PLM0?-
&?#
!?
inputs??????????
? "???????????
E__inference_dense_30_layer_call_and_return_conditional_losses_1085874]23/?,
%?"
 ?
inputs?????????d
? "&?#
?
0??????????
? ~
*__inference_dense_30_layer_call_fn_1085864P23/?,
%?"
 ?
inputs?????????d
? "????????????
E__inference_dense_31_layer_call_and_return_conditional_losses_1085983^890?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_31_layer_call_fn_1085973Q890?-
&?#
!?
inputs??????????
? "????????????
F__inference_flatten_7_layer_call_and_return_conditional_losses_1086327a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
+__inference_flatten_7_layer_call_fn_1086321T7?4
-?*
(?%
inputs?????????@
? "????????????
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1086287h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
0__inference_leaky_re_lu_21_layer_call_fn_1086282[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1086316h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
0__inference_leaky_re_lu_22_layer_call_fn_1086311[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1086356Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
0__inference_leaky_re_lu_23_layer_call_fn_1086351M0?-
&?#
!?
inputs??????????
? "????????????
F__inference_reshape_7_layer_call_and_return_conditional_losses_1086092a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????@
? ?
+__inference_reshape_7_layer_call_fn_1086078T0?-
&?#
!?
inputs??????????
? " ??????????@?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084364{FGHIJKLMH?E
>?;
1?.
conv2d_14_input?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1084392{FGHIJKLMH?E
>?;
1?.
conv2d_14_input?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1085821rFGHIJKLM??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1085855rFGHIJKLM??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_21_layer_call_fn_1084181nFGHIJKLMH?E
>?;
1?.
conv2d_14_input?????????
p 

 
? "???????????
/__inference_sequential_21_layer_call_fn_1084336nFGHIJKLMH?E
>?;
1?.
conv2d_14_input?????????
p

 
? "???????????
/__inference_sequential_21_layer_call_fn_1085766eFGHIJKLM??<
5?2
(?%
inputs?????????
p 

 
? "???????????
/__inference_sequential_21_layer_call_fn_1085787eFGHIJKLM??<
5?2
(?%
inputs?????????
p

 
? "???????????
J__inference_sequential_22_layer_call_and_return_conditional_losses_1084004?23746589=:<;>?@ABCDE??<
5?2
(?%
dense_30_input?????????d
p 

 
? "-?*
#? 
0?????????
? ?
J__inference_sequential_22_layer_call_and_return_conditional_losses_1084060?23674589<=:;>?@ABCDE??<
5?2
(?%
dense_30_input?????????d
p

 
? "-?*
#? 
0?????????
? ?
J__inference_sequential_22_layer_call_and_return_conditional_losses_1085597~23746589=:<;>?@ABCDE7?4
-?*
 ?
inputs?????????d
p 

 
? "-?*
#? 
0?????????
? ?
J__inference_sequential_22_layer_call_and_return_conditional_losses_1085739~23674589<=:;>?@ABCDE7?4
-?*
 ?
inputs?????????d
p

 
? "-?*
#? 
0?????????
? ?
/__inference_sequential_22_layer_call_fn_1083707y23746589=:<;>?@ABCDE??<
5?2
(?%
dense_30_input?????????d
p 

 
? " ???????????
/__inference_sequential_22_layer_call_fn_1083948y23674589<=:;>?@ABCDE??<
5?2
(?%
dense_30_input?????????d
p

 
? " ???????????
/__inference_sequential_22_layer_call_fn_1085438q23746589=:<;>?@ABCDE7?4
-?*
 ?
inputs?????????d
p 

 
? " ???????????
/__inference_sequential_22_layer_call_fn_1085483q23674589<=:;>?@ABCDE7?4
-?*
 ?
inputs?????????d
p

 
? " ???????????
J__inference_sequential_23_layer_call_and_return_conditional_losses_1084824?23746589=:<;>?@ABCDEFGHIJKLMD?A
:?7
-?*
sequential_22_input?????????d
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_23_layer_call_and_return_conditional_losses_1084886?23674589<=:;>?@ABCDEFGHIJKLMD?A
:?7
-?*
sequential_22_input?????????d
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_23_layer_call_and_return_conditional_losses_1085158~23746589=:<;>?@ABCDEFGHIJKLM7?4
-?*
 ?
inputs?????????d
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_23_layer_call_and_return_conditional_losses_1085330~23674589<=:;>?@ABCDEFGHIJKLM7?4
-?*
 ?
inputs?????????d
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_23_layer_call_fn_1084517~23746589=:<;>?@ABCDEFGHIJKLMD?A
:?7
-?*
sequential_22_input?????????d
p 

 
? "???????????
/__inference_sequential_23_layer_call_fn_1084762~23674589<=:;>?@ABCDEFGHIJKLMD?A
:?7
-?*
sequential_22_input?????????d
p

 
? "???????????
/__inference_sequential_23_layer_call_fn_1084953q23746589=:<;>?@ABCDEFGHIJKLM7?4
-?*
 ?
inputs?????????d
p 

 
? "???????????
/__inference_sequential_23_layer_call_fn_1085014q23674589<=:;>?@ABCDEFGHIJKLM7?4
-?*
 ?
inputs?????????d
p

 
? "???????????
%__inference_signature_wrapper_1085393?23746589=:<;>?@ABCDEFGHIJKLMS?P
? 
I?F
D
sequential_22_input-?*
sequential_22_input?????????d"=?:
8
sequential_21'?$
sequential_21?????????