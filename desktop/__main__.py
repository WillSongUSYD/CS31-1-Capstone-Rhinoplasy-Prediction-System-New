"""Enable ``python -m desktop`` as an alternate entrypoint.

``python -m desktop`` is slightly more idiomatic than
``python desktop/app.py`` because Python then sets up sys.path properly
for our sibling imports, and py2app's ``setup(app=['desktop/app.py'])``
pointer still works the same way at bundle time.
"""
import sys

from desktop.app import main

if __name__ == "__main__":
    sys.exit(main())
