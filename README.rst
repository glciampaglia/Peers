==============================
The `Peers` agent-based model.
==============================

------------
Requirements
------------
See requirements.txt

-----------------
Package structure
-----------------

`peers` - root package

- `peers.design` - experimental designs (LHD, winding stairs)
- `peers.gsa` - global sensitivity analysis
- `peers.fit` - model calibration
- `peers.mde` - model calibration via minimum distance estimation (does not work)
- `peers.tests` - tests (requires nose)

----------------
Invocation
----------------

Use `peerstool` to execute the command you want. `peerstool -h` will print the
list of available commands::

$ peerstool <cmd> -h

will print the help message of command `<cmd>`. Each command is implemented in
its own submodule, and submodules of related commands are all in the same
subpackage. Note that most submodules are also standalone scripts themselves!
Thus, if you have `peers` in you `PYTHONPATH`, then you can always call the
submodule explicity. For example, command `sigmoid` is implemented in submodule
`peers.fit.sigmoid` of subpackage `peers.fit`. Then you can execute it via:: 

$ python -m peers.fit.sigmoid

You can inspect the source code of `peerstool` to figure out the location of
any command (look for a dict variable named `commands`).

