### Paso de PyQt5 .ui a un script de Python

Para convertir el archivo .ui que genera QtDesigner en un script de Python, primero se debe instalar pyuic5 con la siguiente línea en consola

```bash
sudo apt install pyqt5-dev-tools
```

y luego ejecutar la siguiente línea.

```bash
pyuic5 -x GUI/user-interface.ui -o user-interface.py
```