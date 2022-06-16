# Prototype Litfin
Prototype repository to provide a proof of concept for the Litfin python
package.

# PIP install from github
Install this package from git using pip:
```
# Create virtual environment
python3 -m venv .venv
# Activate virtual environment
.venv/bin/activate
# pip install from git
python3 -m pip install git+https://github.com/FridayFawkes/proto_litfin.git@master
```
## Example Code
Example usage:
```Python
import proto_litfin as lf
# Download data
etfs = lf.download_quotes_yahoo(["IWDA.AS", "VWCE.DE"], growth_index=True, start_date="2000-01-01")

# Comparison graph
lf.ichart(etfs_yahoo_finance, yticksuffix="€", title="Evolução de cada 100 €uros investidos").show()
```

Check more examples
[here](https://github.com/FridayFawkes/proto_litfin/blob/master/test/test_live002.py)

# Contributing
Quick guide for developing the package.

Check `Makefile` for more details.

- Clone the repository:
```
git clone git@github.com:FridayFawkes/proto_litfin.git
```
- Setup a virtual environment:
```
make create-venv
```
- Build the package:
```
make build
```
- Install in developing virtual environment:
```
make install-package
```
- Run tests with installed package:
```
make test
```
- Reset environment:
```
make clean-all
```
