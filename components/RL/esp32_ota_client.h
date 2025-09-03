// esp32_ota_client.h
#include <WiFi.h>
#include <HTTPClient.h>
#include <Update.h>
#include <ArduinoJson.h>

class ESP32OTAClient {
private:
    String deviceId;
    String currentVersion;
    String otaServerUrl;
    
public:
    ESP32OTAClient(String serverUrl, String devId) {
        otaServerUrl = serverUrl;
        deviceId = devId;
        currentVersion = "0.9.0"; // 当前固件版本
    }
    
    bool checkForUpdates() {
        HTTPClient http;
        String url = otaServerUrl + "/api/check-update/" + deviceId + "/" + currentVersion;
        
        http.begin(url);
        int httpCode = http.GET();
        
        if (httpCode == 200) {
            String payload = http.getString();
            DynamicJsonDocument doc(1024);
            deserializeJson(doc, payload);
            
            bool updateAvailable = doc["update_available"];
            if (updateAvailable) {
                Serial.println("Update available! Downloading...");
                return downloadAndInstallUpdate();
            }
        }
        return false;
    }
    
    bool downloadAndInstallUpdate() {
        HTTPClient http;
        http.begin(otaServerUrl + "/api/download-update");
        
        int httpCode = http.GET();
        if (httpCode == 200) {
            int contentLength = http.getSize();
            WiFiClient* stream = http.getStreamPtr();
            
            if (Update.begin(contentLength)) {
                size_t written = Update.writeStream(*stream);
                
                if (written == contentLength) {
                    Serial.println("Written : " + String(written) + " successfully");
                } else {
                    Serial.println("Written only : " + String(written) + "/" + String(contentLength));
                    return false;
                }
                
                if (Update.end()) {
                    Serial.println("OTA done!");
                    if (Update.isFinished()) {
                        Serial.println("Update successfully completed. Rebooting...");
                        ESP.restart();
                        return true;
                    }
                }
            }
        }
        return false;
    }
};