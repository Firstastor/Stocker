import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3 
import QtQuick.Layouts
import "StockPage"

Page{
    id: root
    ColumnLayout {
        anchors.fill: parent
        spacing: 5

        TabBar{
            id: tabBar
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            currentIndex: 0
            
            TabButton {
                text: qsTr("股票信息")
                onClicked: tabBar.currentIndex = 0
            }
            TabButton {
                text: qsTr("股票图表")
                onClicked: tabBar.currentIndex = 1
            }

            TabButton {
                text: qsTr("股票预测")
                onClicked: tabBar.currentIndex = 2
            }
        }
        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: tabBar.currentIndex  

            StockInfoPage {
                id: stockInfoPage
                onStockSelected: function(code, name) {
                    tabBar.currentIndex = 1
                    stockPlotPage.setStockData(code, name)
                }
            }
            StockPlotPage {
                id: stockPlotPage
            }

            StockForecastPage {
                id: stockForecastPage
            }
        }
    }
}