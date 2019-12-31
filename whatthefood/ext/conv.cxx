#include <Python.h>

#include <numpy/npy_no_deprecated_api.h>
#include <numpy/arrayobject.h>


int _conv2d_check_args(PyArrayObject* arr, PyArrayObject* filters, int step)
{
	if (PyArray_NDIM(arr) != 4)
	{
		PyErr_SetString(PyExc_ValueError, "Invalid input array shape");
		return 0;
	}

	if (PyArray_NDIM(filters) != 4)
	{
		PyErr_SetString(PyExc_ValueError, "Invalid filters array shape");
		return 0;
	}

	if (PyArray_TYPE(arr) != NPY_FLOAT32)
	{
		PyErr_SetString(PyExc_ValueError, "Invalid input array type - use float32");
		return 0;
	}

	if (PyArray_TYPE(filters) != NPY_FLOAT32)
	{
		PyErr_SetString(PyExc_ValueError, "Invalid filters array type - use float32");
		return 0;
	}

	if (step <= 0)
	{
		PyErr_SetString(PyExc_ValueError, "Invalid step");
		return 0;
	}

	return 1;
}

int _conv2d_grad_check_args(PyArrayObject* arr, PyArrayObject* filters, PyArrayObject* grad, int step)
{
	if (!_conv2d_check_args(arr, filters, step))
		return 0;

	if (PyArray_NDIM(arr) != 4)
	{
		PyErr_SetString(PyExc_ValueError, "Invalid grad array shape");
		return 0;
	}
	if (PyArray_DIM(arr, 0) != PyArray_DIM(grad, 0))
	{
		PyErr_SetString(PyExc_ValueError, "Invalid grad array dim 0");
		return 0;
	}
	if (PyArray_DIM(filters, 3) != PyArray_DIM(grad, 3))
	{
		PyErr_SetString(PyExc_ValueError, "Invalid grad array dim 3");
		return 0;
	}

	return 1;
}

PyArrayObject* _conv2d_build_output(PyArrayObject* arr, PyArrayObject* filters, int step, int padding_same)
{
	npy_intp dims[4] = {
		PyArray_DIM(arr, 0),
		padding_same ? (PyArray_DIM(arr, 1) - 1) / step + 1: (PyArray_DIM(arr, 1) - PyArray_DIM(filters, 0)) / step + 1,
		padding_same ? (PyArray_DIM(arr, 1) - 1) / step + 1 : (PyArray_DIM(arr, 2) - PyArray_DIM(filters, 1)) / step + 1,
		PyArray_DIM(filters, 3)
	};

	if (dims[1] <= 0 || dims[2] <= 0)
	{
		PyErr_SetString(PyExc_ValueError, "Output array shape would have non-positive dimensions");
		return NULL;
	}

	return (PyArrayObject*)PyArray_SimpleNew(4, dims, PyArray_TYPE(arr));
}

void _conv2d_run_batched(PyArrayObject* out, PyArrayObject* arr, PyArrayObject* filters, int step, int padding_same)
{
//#pragma omp paralell for
	for (npy_intp sample = 0; sample < PyArray_DIM(out, 0); ++sample)
	{
		npy_intp isi = sample * PyArray_STRIDE(arr, 0) / sizeof(float);
		npy_intp osi = sample * PyArray_STRIDE(out, 0) / sizeof(float);
		for (npy_intp ox = 0; ox < PyArray_DIM(out, 1); ++ox)
		{
			npy_intp oxi = ox * PyArray_STRIDE(out, 1) / sizeof(float);

			npy_intp minx = 0;
			npy_intp maxx = PyArray_DIM(filters, 0);

			if (padding_same)
			{
				npy_intp x0 = step * ox - maxx / 2;
				if (x0 < 0)
					minx += x0;
				if (x0 + maxx > PyArray_DIM(arr, 1))
					maxx = PyArray_DIM(arr, 1) - x0;
			}

			for (npy_intp oy = 0; oy < PyArray_DIM(out, 2); ++oy)
			{
				npy_intp oyi = oy * PyArray_STRIDE(out, 2) / sizeof(float);

				npy_intp miny = 0;
				npy_intp maxy = PyArray_DIM(filters, 1);

				if (padding_same)
				{
					npy_intp y0 = step * oy - maxy / 2;
					if (y0 < 0)
						miny += y0;
					if (y0 + maxy > PyArray_DIM(arr, 2))
						maxy = PyArray_DIM(arr, 2) - y0;
				}

				for (npy_intp oc = 0; oc < PyArray_DIM(out, 3); ++oc)
				{
					float value = 0;

					for (npy_intp x = minx; x < maxx; ++x)
					{
						npy_intp ix = step * ox + x;
						if (padding_same)
							ix -= PyArray_DIM(filters, 0) / 2;
						npy_intp ixi = ix * PyArray_STRIDE(arr, 1) / sizeof(float);
						npy_intp ixf = x * PyArray_STRIDE(filters, 0) / sizeof(float);

						for (npy_intp y = miny; y < maxy; ++y)
						{
							npy_intp iy = step * oy + y;
							if (padding_same)
								iy -= PyArray_DIM(filters, 1) / 2;
							npy_intp iyi = iy * PyArray_STRIDE(arr, 2) / sizeof(float);
							npy_intp iyf = y * PyArray_STRIDE(filters, 1) / sizeof(float);

							for (npy_intp ic = 0; ic < PyArray_DIM(arr, 3); ++ic)
							{
								npy_intp icf = ic * PyArray_STRIDE(filters, 2) / sizeof(float);
								value += ((float*)PyArray_DATA(arr))[isi + ixi + iyi + ic] 
									* ((float*)PyArray_DATA(filters))[ixf + iyf + icf + oc];
							}
						}
					}

					((float*)PyArray_DATA(out))[osi + oxi + oyi + oc] = value;
				}
			}
		}
	}
}

PyObject* _conv2d_impl(PyArrayObject* arr, PyArrayObject* filters, int step, int padding_same)
{
	if (!_conv2d_check_args(arr, filters, step))
		return NULL;

	PyArrayObject* output = _conv2d_build_output(arr, filters, step, padding_same);
	if (output == NULL)
		return NULL;

	_conv2d_run_batched(output, arr, filters, step, padding_same);

	return (PyObject*)output;
}

void _conv2d_grad_run(PyArrayObject* arr_grad, PyArrayObject* filters_grad, PyArrayObject* arr, PyArrayObject* filters, PyArrayObject* grad, int step, int padding_same)
{
//#pragma omp parallel for
	for (npy_intp ic = 0; ic < PyArray_DIM(arr, 3); ++ic)
	{
		npy_intp icf = ic * PyArray_STRIDE(filters, 2) / sizeof(float);
		for (npy_intp sample = 0; sample < PyArray_DIM(grad, 0); ++sample)
		{
			npy_intp isi = sample * PyArray_STRIDE(arr, 0) / sizeof(float);
			npy_intp osi = sample * PyArray_STRIDE(grad, 0) / sizeof(float);
			for (npy_intp ox = 0; ox < PyArray_DIM(grad, 1); ++ox)
			{
				npy_intp oxi = ox * PyArray_STRIDE(grad, 1) / sizeof(float);

				npy_intp minx = 0;
				npy_intp maxx = PyArray_DIM(filters, 0);

				if (padding_same)
				{
					npy_intp x0 = step * ox - maxx / 2;
					if (x0 < 0)
						minx += x0;
					if (x0 + maxx > PyArray_DIM(arr, 1))
						maxx = PyArray_DIM(arr, 1) - x0;
				}

				for (npy_intp oy = 0; oy < PyArray_DIM(grad, 2); ++oy)
				{
					npy_intp oyi = oy * PyArray_STRIDE(grad, 2) / sizeof(float);

					npy_intp miny = 0;
					npy_intp maxy = PyArray_DIM(filters, 1);

					if (padding_same)
					{
						npy_intp y0 = step * oy - maxy / 2;
						if (y0 < 0)
							miny += y0;
						if (y0 + maxy > PyArray_DIM(arr, 2))
							maxy = PyArray_DIM(arr, 2) - y0;
					}

					for (npy_intp oc = 0; oc < PyArray_DIM(grad, 3); ++oc)
					{
						float g = ((float*)PyArray_DATA(grad))[osi + oxi + oyi + oc];

						for (npy_intp x = minx; x < maxx; ++x)
						{
							npy_intp ix = step * ox + x;
							if (padding_same)
								ix -= PyArray_DIM(filters, 0) / 2;
							npy_intp ixi = ix * PyArray_STRIDE(arr, 1) / sizeof(float);
							npy_intp ixf = x * PyArray_STRIDE(filters, 0) / sizeof(float);

							for (npy_intp y = miny; y < maxy; ++y)
							{
								npy_intp iy = step * oy + y;
								if (padding_same)
									iy -= PyArray_DIM(filters, 1) / 2;
								npy_intp iyi = iy * PyArray_STRIDE(arr, 2) / sizeof(float);
								npy_intp iyf = y * PyArray_STRIDE(filters, 1) / sizeof(float);

								((float*)PyArray_DATA(arr_grad))[isi + ixi + iyi + ic] += ((float*)PyArray_DATA(filters))[ixf + iyf + icf + oc] * g;
								((float*)PyArray_DATA(filters_grad))[ixf + iyf + icf + oc] += ((float*)PyArray_DATA(arr))[isi + ixi + iyi + ic] * g;
							}
						}
					}
				}
			}
		}
	}
}

PyObject* _conv2d_grad_impl(PyArrayObject* arr, PyArrayObject* filters, PyArrayObject* grad, int step, int padding_same)
{
	if (!_conv2d_grad_check_args(arr, filters, grad, step))
		return NULL;

	PyArrayObject* arr_grad = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(arr), PyArray_DIMS(arr), PyArray_TYPE(arr), 0);
	PyArrayObject* filters_grad = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(filters), PyArray_DIMS(filters), PyArray_TYPE(filters), 0);

	_conv2d_grad_run(arr_grad, filters_grad, arr, filters, grad, step, padding_same);

	PyObject* ret = PyTuple_New(2);
	PyTuple_SetItem(ret, 0, (PyObject*)arr_grad);
	PyTuple_SetItem(ret, 1, (PyObject*)filters_grad);

	return ret; //Instead of PyTuple_Pack to keep correct reference count for returned objects
}

static PyObject* conv2d(PyObject* dummy, PyObject* args)
{
	PyObject* arr = NULL;
	PyObject* filters = NULL;
	int step;
	int padding_same;

	if (!PyArg_ParseTuple(args, "O!O!ip", &PyArray_Type, &arr, &PyArray_Type, &filters, &step, &padding_same))
		return NULL;

	return _conv2d_impl((PyArrayObject*)arr, (PyArrayObject*)filters, step, padding_same);
}

static PyObject* conv2d_grad(PyObject* dummy, PyObject* args)
{
	PyObject* arr = NULL;
	PyObject* filters = NULL;
	PyObject* grad = NULL;
	int step;
	int padding_same;

	if (!PyArg_ParseTuple(args, "O!O!O!ip", &PyArray_Type, &arr, &PyArray_Type, &filters, &PyArray_Type, &grad, &step, &padding_same))
		return NULL;

	return _conv2d_grad_impl((PyArrayObject*)arr, (PyArrayObject*)filters, (PyArrayObject*)grad, step, padding_same);
}

static PyMethodDef ConvMethods[] = {
	{"conv2d", conv2d, METH_VARARGS, "2d convolution"},
	{"conv2d_grad", conv2d_grad, METH_VARARGS, "2d convolution gradient"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef convmodule = {
	PyModuleDef_HEAD_INIT,
	"conv",
	NULL,
	-1,
	ConvMethods
};

PyMODINIT_FUNC
PyInit_conv(void)
{
	if (PyArray_API == NULL)
	{
		import_array();
	}
	return PyModule_Create(&convmodule);
}
