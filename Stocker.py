import pandas as pd
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from Src.Backend.Stock import StockCalculate, StockGet, StockUpdater
import sys

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)

    stock_calculate = StockCalculate()
    stock_get = StockGet()

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("StockCalculate", stock_calculate)
    engine.rootContext().setContextProperty("StockGet", stock_get)
    engine.load("Src/Frontend/Main.qml")

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec())