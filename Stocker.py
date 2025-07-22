import pandas as pd
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from Src.Backend.StockInfo import StockInfoGet, StockInfoProcess, StockInfoUpdater
import sys

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)


    stock_info_get = StockInfoGet()
    stock_info_data = StockInfoProcess(stock_info_get.get_stock_data())
    updater = StockInfoUpdater( stock_info_get, stock_info_data)
    updater.start(10000) 

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("StockInfoGet", stock_info_get)
    engine.rootContext().setContextProperty("StockInfoData", stock_info_data)
    engine.load("Src/Frontend/Main.qml")

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec())