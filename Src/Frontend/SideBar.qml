import QtQuick
import QtQuick.Controls 
import QtQuick.Controls.FluentWinUI3 
import QtQuick.Layouts
import QtQuick.Window 

Rectangle {

    color: palette.window

    signal switchPage(int index)
    property int currentIndex: 0

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        Button {
            id: pushbtndownload
            Layout.alignment: Qt.AlignCenter
            icon.source: "../Image/download.png"
            display: AbstractButton.TextUnderIcon
            text: qsTr("股票")
            flat: true
            highlighted: parent.parent.currentIndex === 0
            onClicked: {
                parent.parent.currentIndex = 0
                switchPage(0)
            }
        }
        Item {
            Layout.fillHeight: true
            Layout.fillWidth: true
        }
        Button {
            id: pushbtnsetting
            Layout.alignment: Qt.AlignCenter
            icon.source: "../Image/setting.png"
            display: AbstractButton.TextUnderIcon
            text: qsTr("设置")
            flat: true
            highlighted: parent.parent.currentIndex === 1
            onClicked: {
                parent.parent.currentIndex = 1
                switchPage(1)
            }
        }
    }
}