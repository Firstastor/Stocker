import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3 
import QtQuick.Layouts
import "StockPlotPageComponent"

Page {
    id: root
    property string stockCode: ""
    property string stockName: ""
    property var stockData: []
    property var historyData: []
    property int currentScale: stockHeader.currentScale
    property int totalYears: 10
    property alias showMacd: stockChart.showMacd
    property alias showRsi: stockChart.showRsi
    property alias showKdj: stockChart.showKdj
    
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
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: palette.mid
        }
        StockHeader {
            id: stockHeader
            stockCode: root.stockCode
            stockName: root.stockName
            Layout.fillWidth: true
            stockData: root.stockData
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
        }


        Loader {
            id: subChartLoader
            Layout.preferredWidth: parent.width - 60
            Layout.preferredHeight: (root.showMacd || root.showRsi || root.showKdj)  ? root.height*0.2 : 0
            Layout.alignment: Qt.AlignCenter
            Behavior on Layout.preferredHeight {
                NumberAnimation {
                    duration: 200  
                    easing.type: Easing.InOutQuad  
                }
            }
            sourceComponent: {
                if (root.showMacd) return macdComponent
                if (root.showRsi) return rsiComponent
                if (root.showKdj) return kdjComponent
                return null
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