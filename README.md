# DREAM

The code is based on MPI whose installation can be done followed by this guide at https://ireneli.eu/2016/02/15/installation/.

Run the code with

```
mpiexec -n 8 python demo.py
```
The datasets used in our experiments are available at https://www.csie.ntu.edu.tw/~cjlin/libsvm/.

You can change different datasets by change the `default={%dataset}` in the following code in  `demo.py`
```
parser.add_argument('--data_file', default='ijcnn1', type=str, help='file of data')
```
We use dataset a9a, w8a and ijcnn1 in our experiments.
