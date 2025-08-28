import json

# Sample message (would normally come from MQTT/HTTP/etc)
message_str = '{"fea_weights":[0.2,0.1,0.3],"fea_label":[1],"client_id":"esp32_8"}'


def parse_message(message_str):
    try:
        # Step 1: Parse JSON string to Python dict
        message = json.loads(message_str)

        # Step 2: Extract and validate values
        fea_weights = message.get('fea_weights', [])
        fea_label = message.get('fea_label', [None])[0]  # Get first element or None
        client_id = message.get('client_id', '')

        # Step 3: Convert to appropriate types
        fea_weights = [float(x) for x in fea_weights]  # Ensure all are floats
        fea_label = int(fea_label) if fea_label is not None else None

        # Step 4: Return structured data
        return {
            'weights': fea_weights,
            'label': fea_label,
            'client_id': client_id  # Keep as string if it's alphanumeric
        }

    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None
    except (ValueError, TypeError) as e:
        print(f"Data conversion error: {e}")
        return None


# Usage
parsed = parse_message(message_str)
if parsed:
    print("Weights:", parsed['weights'][:5], "...")  # Show first 5 weights
    print("Label:", parsed['label'])
    print("Client ID:", parsed['client_id'])