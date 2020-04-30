Contributing
============
All development is done on GitHub: https://github.com/kumuji/volumentations

If you find a bug or have a feature request file an issue at https://github.com/kumuji/volumentations/issues

To create a pull request:

1. Fork the repository.
2. Clone it.
3. Install development requirements:

.. code-block:: bash

    python -m pip install poetry
    poetry install --dev

4. Initialize it from the folder with the repo:

.. code-block:: bash

    pre-commit install


4. Make desired changes to the code.
5. Check if your changes passing the tests:


.. code-block:: bash

    nox -rs

7. Push code to your forked repo.
8. Create pull request.
