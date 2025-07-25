import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3 
import QtQuick.Layouts
import QtQuick.Window  

Page {
    id: root
    property string sortField: "代码"
    property bool sortAscending: true
    property string filterString: ""
    property var stockData: []
    
    signal stockSelected(string code, string name)
    
    function updateSort(field) {
        if (sortField === field) {
            sortAscending = !sortAscending
        } else {
            sortField = field
            sortAscending = true
        }
        applySortAndFilter()
    }
    
    function applySortAndFilter() {
        var filtered = StockGet.filter_stock_data(searchField.text)
        stockData = StockGet.sort_stock_data(filtered, sortField, sortAscending)
    }
    
    Component.onCompleted: {
        // Initialize with all stock data
        stockData = StockGet.get_stock_data()
    }
    
    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // 搜索框
        TextField {
            id: searchField
            Layout.fillWidth: true
            placeholderText: qsTr("搜索股票代码或名称...")
            selectByMouse: true
            leftPadding: 10
            rightPadding: 10
            topPadding: 8
            bottomPadding: 8
            background: Rectangle {
                color: "transparent"
                border.color: palette.mid
                radius: 4
            }
            onTextChanged: {
                applySortAndFilter()
            }
        }

        // 表头
        RowLayout {
            id: headerRow
            Layout.fillWidth: true
            spacing: 1
            height: 40

            HeaderButton {
                text: qsTr("代码")
                Layout.preferredWidth: root.width/5 
                sortField: "代码"
                currentSortField: root.sortField
                ascending: root.sortAscending
                onClicked: updateSort("代码")
            }

            HeaderButton {
                text: qsTr("名称")
                Layout.preferredWidth: root.width/5
                sortField: "名称"
                currentSortField: root.sortField
                ascending: root.sortAscending
                onClicked: updateSort("名称")
            }

            HeaderButton {
                text: qsTr("最新价")
                Layout.preferredWidth: root.width/5
                sortField: "最新价"
                currentSortField: root.sortField
                ascending: root.sortAscending
                onClicked: updateSort("最新价")
            }

            HeaderButton {
                text: qsTr("涨幅")
                Layout.preferredWidth: root.width/5
                sortField: "涨幅"
                currentSortField: root.sortField
                ascending: root.sortAscending
                onClicked: updateSort("涨幅")
            }

            HeaderButton {
                text: qsTr("涨跌")
                Layout.preferredWidth: root.width/5
                sortField: "涨跌"
                currentSortField: root.sortField
                ascending: root.sortAscending
                onClicked: updateSort("涨跌")
            }
        }

        // 股票列表
        ListView {
            id: listView
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 1
            clip: true
            model: stockData
            cacheBuffer: 400
            boundsBehavior: Flickable.StopAtBounds
            maximumFlickVelocity: 1500
            flickDeceleration: 2000
            
            delegate: Item {
                id: delegateItem
                width: ListView.view.width
                height: 40

                Rectangle {
                    id: contentItem
                    anchors.fill: parent
                    color: mouseArea.containsMouse ? Qt.lighter(palette.highlight, 1.5) : palette.base
                    border.width: 1
                    border.color: palette.mid
                    Behavior on color { ColorAnimation { duration: 150 } }

                    MouseArea {
                        id: mouseArea
                        anchors.fill: parent
                        hoverEnabled: true
                        onClicked: root.stockSelected(modelData.代码, modelData.名称)
                    }

                    RowLayout {
                        anchors.fill: parent
                        spacing: 1

                        Text {
                            text: modelData.代码
                            color: mouseArea.containsMouse ? palette.highlightedText : palette.text
                            Layout.preferredWidth: root.width/5
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            font.pixelSize: 12
                        }

                        Text {
                            text: modelData.名称
                            color: mouseArea.containsMouse ? palette.highlightedText : palette.text
                            Layout.preferredWidth: root.width/5
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            font.pixelSize: 12
                        }

                        Text {
                            text: modelData.最新价.toFixed(2)
                            Layout.preferredWidth: root.width/5
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            font.pixelSize: 12
                            font.bold: true
                            color: getPriceColor(modelData.涨跌, mouseArea.containsMouse)
                        }

                        Text {
                            text: modelData.涨幅.toFixed(2) + "%"
                            Layout.preferredWidth: root.width/5
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            font.pixelSize: 12
                            font.bold: true
                            color: getPriceColor(modelData.涨幅, mouseArea.containsMouse)
                        }

                        Text {
                            text: modelData.涨跌.toFixed(2)
                            Layout.preferredWidth: root.width/5
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            font.pixelSize: 12
                            font.bold: true
                            color: getPriceColor(modelData.涨跌, mouseArea.containsMouse)
                        }
                    }
                }
            }

            Label {
                anchors.centerIn: parent
                visible: listView.count === 0
                text: qsTr("没有找到匹配的股票")
                font.pixelSize: 16
                color: palette.mid
            }
        }
    }

    component HeaderButton : Button {
        property string sortField: ""
        property string currentSortField: ""
        property bool ascending: true
        flat: true
        font.bold: true
        readonly property string displayText: {
            if (sortField === currentSortField) {
                return text + (ascending ? " ↑" : " ↓")
            }
            return text
        }
        contentItem: Text {
            text: parent.displayText
            font: parent.font
            color: parent.down ? palette.highlight : palette.text
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
        background: Rectangle {
            color: parent.down ? Qt.darker(palette.mid, 1.2) : 
                parent.hovered ? Qt.lighter(palette.mid, 1.5) : palette.mid
            radius: 2
        }
    }

    function getPriceColor(value, isHovered) {
        if (value > 0) {
            return isHovered ? "#aaffaa" : "#00aa00"
        } else if (value < 0) {
            return isHovered ? "#ffaaaa" : "#aa0000"
        }
        return isHovered ? palette.highlightedText : palette.text
    }
}