import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3 
import QtQuick.Layouts


Page {
    id: root
    property string stockCode: ""
    property string stockName: ""
    property var historyData: []
    
    // 与主页面连接
    function setStockData(code, name) {
        stockCode = code
        stockName = name
        loadHistoryData()
    }
    
    // 加载历史数据
    function loadHistoryData() {
        historyData = StockGet.get_history_stock_data(stockCode, 240, 365*3) // 3年日线数据
    }
    

}