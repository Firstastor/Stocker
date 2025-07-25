import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3 
import QtQuick.Layouts
import "StockPage"

Page {
    id: root
    property bool stockSelected: false
    property string selectedStockCode: ""
    property string selectedStockName: ""

    Item {
        anchors.fill: parent

        Item {
            id: leftPanel
            width: stockSelected ? 350 : parent.width
            height: parent.height
            anchors.left: parent.left

            StockInfoPage {
                id: stockInfoPage
                anchors.fill: parent
                onStockSelected: function(code, name) {
                    root.stockSelected = true
                    root.selectedStockCode = code
                    root.selectedStockName = name
                    stockPlotPage.setStockData(code, name)
                    stockPredicationPage.setStockData(code, name)
                }
            }

            Behavior on width {
                NumberAnimation { duration: 200 }
            }
        }

        Item {
            id: rightPanel
            width: stockSelected ? parent.width - leftPanel.width : 0
            height: parent.height
            anchors.right: parent.right
            clip: true

            Column {
                anchors.fill: parent
                spacing: 5

                StockPlotPage {
                    id: stockPlotPage
                    width: parent.width
                    height: parent.height * 0.5
                }

                StockPredicationPage {
                    id: stockPredicationPage
                    width: parent.width
                    height: parent.height * 0.5
                }
            }
        }
    }
}