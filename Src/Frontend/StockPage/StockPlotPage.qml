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

    // 颜色定义
    property color upColor: "red"
    property color downColor: "green"
    property color bgColor: palette.base

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
        RowLayout {
            Layout.fillWidth: true

            // 股票标题栏
            StockHeader {
                id: stockHeader
                stockCode: root.stockCode
                stockName: root.stockName
                bgColor: root.bgColor
                upColor: root.upColor
                Layout.fillWidth: true
            }

            ScaleControls {
                Layout.fillWidth: true
                onScaleSelected: function(newScale) {
                    root.currentScale = newScale
                    root.refreshData() 
                }
            }
        }

        // K线图主体
        KLineChart {
            id: klineChart
            historyData: root.historyData
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }
}