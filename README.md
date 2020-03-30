# Instructions

1. Install METIS 5.1.0, HDF5 and Legion built with GasNET in Release mode.
2. Modify `configure.sh` to point to the right directories.
3. `mkdir build; cd build; ../configure.sh; make install`
4. Go to the `run` directory to run a case. Modify the configuration in `run.sh` and use the python script `debug.py` with the correct expected error.