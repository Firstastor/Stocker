import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3 
import QtQuick.Layouts

Window {
    id: mainWindow
    width: 1080
    height: 720
    title: qsTr("Stocker")
    visible: true
    flags: Qt.Window | Qt.FramelessWindowHint
    color: "transparent"

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

            StackLayout {
                id: stackLayout
                Layout.fillWidth: true
                Layout.fillHeight: true
                
                StockPage {
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