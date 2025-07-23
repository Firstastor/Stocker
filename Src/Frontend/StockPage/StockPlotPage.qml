import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3 
import QtQuick.Layouts
import "StockPlotPageComponent"

Page {
    id: root
    property string stockCode: ""
    property string stockName: ""
    property var historyData: []
    property int currentScale: stockHeader.currentScale
    property int totalYears: 5


    function setStockData(code, name) {
        stockCode = code
        stockName = name
        refreshData()
    }

    function refreshData() {
        historyData = StockInfoGet.get_history_stock_data(stockCode, currentScale, totalYears * 365)
        stockHeader.updateStockData(stockCode, stockName)
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        StockHeader {
            id: stockHeader
            stockCode: root.stockCode
            stockName: root.stockName
            Layout.fillWidth: true
            onScaleSelected: function(newScale) {
                root.currentScale = newScale
                root.refreshData()
            }
        }



        // K线图主体
        StockChart {
            id: stockChart
            historyData: root.historyData
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }
}