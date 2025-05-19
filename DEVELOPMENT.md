## Development Guide / Notes




## Publishing
First update the version numbers in the `setup.py` file and others! Then
```bash
python -m build
```
then
```bash
python patch_wheel.py
```
then
```bash
 python -m twine upload dist/beamz-0.1.0-py3-none-any.whl   
```
(though with the correct version) in order to publis the newest version of the package to pypi.