import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3 
import QtQuick.Layouts

Window {
    id: mainWindow
    width: 1600
    height: 900
    title: qsTr("Stocker")
    visible: true
    flags: Qt.Window | Qt.FramelessWindowHint
    color: "transparent"
    visibility: Window.FullScreen
    
    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        TitleBar {
            id: titleBar
            Layout.fillWidth: true
            Layout.preferredHeight: 40
        }

        RowLayout{
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            SideBar {
                id: sideBar
                Layout.preferredWidth: 120
                Layout.fillHeight: true
                onSwitchPage: function(index) {
                    if (index === 0) {
                        stockPage.stockInfoPage.currentIndex = 0
                    } else if (index === 1) {
                        stockPage.stockInfoPage.currentIndex = 1
                    }
                }
            }

            StackLayout {
                id: stackLayout
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: 0
                
                StockPage {
                    id: stockPage
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
                
                SettingPage {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }
            }
        }
    }
}