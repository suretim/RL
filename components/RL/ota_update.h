// ota_update.h
#ifndef OTA_UPDATE_H
#define OTA_UPDATE_H

#include <WiFi.h>
#include <HTTPClient.h>
#include <Update.h>

class OTAUpdater {
private:
    const char* ssid;
    const char* password;
    const char* update_url;

public:
    OTAUpdater(const char* ssid, const char* password, const char* url)
        : ssid(ssid), password(password), update_url(url) {}

    bool connectWiFi() {
        WiFi.begin(ssid, password);
        int attempts = 0;
        while (WiFi.status() != WL_CONNECTED && attempts < 20) {
            delay(500);
            Serial.print(".");
            attempts++;
        }
        return WiFi.status() == WL_CONNECTED;
    }

    bool updateModel() {
        if (!connectWiFi()) {
            Serial.println("WiFi connection failed");
            return false;
        }

        HTTPClient http;
        http.begin(update_url);
        int httpCode = http.GET();

        if (httpCode == HTTP_CODE_OK) {
            int contentLength = http.getSize();
            if (contentLength > 0) {
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
                            Serial.println("Update successfully completed. Rebooting.");
                            ESP.restart();
                            return true;
                        }
                    } else {
                        Serial.println("Error Occurred. Error #: " + String(Update.getError()));
                    }
                } else {
                    Serial.println("Not enough space to begin OTA");
                }
            }
        } else {
            Serial.println("Failed to download model file");
        }

        http.end();
        return false;
    }
};

#endif