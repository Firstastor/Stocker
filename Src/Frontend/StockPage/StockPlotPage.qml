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
    property int currentSubChart: 0 // 0-none, 1-MACD, 2-RSI, 3-KDJ
    
    function setStockData(code, name) {
        stockCode = code
        stockName = name
        refreshData()
    }

    function refreshData() {
        historyData = StockGet.get_history_stock_data(stockCode, currentScale, totalYears * 365)
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

        StockChart {
            id: stockChart
            historyData: root.historyData
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 300
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 30
            spacing: 5

            ButtonGroup { id: subChartGroup }

            Button {
                text: "无"
                checked: root.currentSubChart === 0
                ButtonGroup.group: subChartGroup
                onClicked: root.currentSubChart = 0
            }
            Button {
                text: "MACD"
                checked: root.currentSubChart === 1
                ButtonGroup.group: subChartGroup
                onClicked: root.currentSubChart = 1
            }
            Button {
                text: "RSI"
                checked: root.currentSubChart === 2
                ButtonGroup.group: subChartGroup
                onClicked: root.currentSubChart = 2
            }
            Button {
                text: "KDJ"
                checked: root.currentSubChart === 3
                ButtonGroup.group: subChartGroup
                onClicked: root.currentSubChart = 3
            }
        }

        Loader {
            id: subChartLoader
            Layout.fillWidth: true
            Layout.preferredHeight: root.currentSubChart > 0 ? 150 : 0
            sourceComponent: {
                if (root.currentSubChart === 1) return macdComponent
                else if (root.currentSubChart === 2) return rsiComponent
                else if (root.currentSubChart === 3) return kdjComponent
                else return null
            }
        }

        Component {
            id: macdComponent
            MACDChart {
                historyData: root.historyData
                visibleData: stockChart.visibleData
                startIndex: stockChart.startIndex
                visibleDays: stockChart.visibleDays
            }
        }

        Component {
            id: rsiComponent
            RSIChart {
                historyData: root.historyData
                visibleData: stockChart.visibleData
                startIndex: stockChart.startIndex
                visibleDays: stockChart.visibleDays
            }
        }

        Component {
            id: kdjComponent
            KDJChart {
                historyData: root.historyData
                visibleData: stockChart.visibleData
                startIndex: stockChart.startIndex
                visibleDays: stockChart.visibleDays
            }
        }
    }
}