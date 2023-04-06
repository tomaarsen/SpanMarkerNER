
Installing SpanMarker
=====================

You may install the `span_marker <https://pypi.org/project/span-marker>`_ Python module via `pip` like so::

    pip install span_marker


PyTorch GPU support
-------------------

If you are installing SpanMarker locally, you may wish to install `torch` in such a way that PyTorch models
can be executed on the GPU. This generally results in large speed improvements, both for training and predicting entities.
I recommend following the official `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

Once installed, you can verify whether `torch` is compiled with CUDA support like so::

    >>> import torch
    >>> torch.cuda.is_available()
    True

And in the context of SpanMarker, you can always move a model to CUDA using::

    >>> model = SpanMarkerModel.from_pretrained(...)
    >>> model.cuda()
    >>> model.device
    device(type='cuda', index=0)