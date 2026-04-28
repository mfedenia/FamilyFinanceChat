import json
import csv
import os
import glob
from datetime import datetime
from collections import defaultdict

def load_user_mapping(csv_path="users.csv"):
    """Load user_id to user info mapping from users.csv"""
    user_map = {}
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_full_path = os.path.join(script_dir, csv_path)
    
    try:
        with open(csv_full_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row['id'].strip('"')
                user_map[uid] = {
                    'name': row['name'],
                    'email': row['email'],
                    'role': row['role']
                }
        print(f"✓ Loaded {len(user_map)} users")
    except FileNotFoundError:
        print(f"⚠ File not found: {csv_full_path}, will use user_id only")
    except Exception as e:
        print(f"⚠ Error reading users.csv: {e}")
    return user_map

def select_json_files():
    """Search for JSON files in current directory and let user select"""
    current_dir = os.getcwd()
    json_files = []
    
    # Search for .json files
    for ext in ["*.json"]:
        json_files.extend(glob.glob(os.path.join(current_dir, ext)))
        json_files.extend(glob.glob(os.path.join(current_dir, "**", ext), recursive=True))
    
    # Remove duplicates and sort
    json_files = sorted(set(json_files))
    
    if not json_files:
        print("No JSON files found in current directory and subdirectories.")
        return None
    
    print(f"\nFound the following JSON files in {current_dir}:")
    print("-" * 60)
    for i, file in enumerate(json_files, 1):
        rel_path = os.path.relpath(file, current_dir)
        file_size = os.path.getsize(file) / 1024  # KB
        print(f"{i:2d}. {rel_path} ({file_size:.1f} KB)")
    
    print("-" * 60)
    print("Please select a file to analyze (enter number):")
    
    while True:
        try:
            user_input = input("Selection: ").strip()
            idx = int(user_input)
            
            if 1 <= idx <= len(json_files):
                selected_file = json_files[idx - 1]
                print(f"Selected: {os.path.relpath(selected_file, current_dir)}")
                return selected_file
            else:
                print(f"Invalid index: {idx}, please enter a number between 1-{len(json_files)}.")
                
        except ValueError:
            print("Invalid input format. Please enter a number.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None

conversations = defaultdict(lambda: defaultdict(list))

try:
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "chat_output")
    
    # Load user mapping
    user_map = load_user_mapping()
    
    # Interactive file selection
    file_name = select_json_files()
    if not file_name:
        exit(0)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        if isinstance(data, list):
            for record in data:
                uid = record.get('user_id', 'unknown')
                conv_id = record.get('id', 'unknown_conversation')
                conv_title = record.get('title', 'Untitled')
                
                messages_dict = record.get('chat', {}).get('history', {}).get('messages', {})
                
                # Collect all messages in this conversation
                conv_messages = []
                for msg_id, msg in messages_dict.items():
                    role = msg.get('role')
                    content = msg.get('content')
                    timestamp = msg.get('timestamp', 0)
                    
                    if role in ('user', 'assistant') and content:
                        conv_messages.append({
                            'role': role,
                            'content': content,
                            'timestamp': timestamp,
                            'date': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d') if timestamp else 'unknown',
                            'time': datetime.fromtimestamp(timestamp).strftime('%H:%M:%S') if timestamp else 'unknown',
                            'conv_id': conv_id,
                            'conv_title': conv_title
                        })
                
                # Sort by timestamp
                conv_messages.sort(key=lambda x: x['timestamp'])
                
                # Group by user and conversation
                if conv_messages:
                    conversations[uid][conv_id] = {
                        'title': conv_title,
                        'messages': conv_messages
                    }
            
            # Output summary with user names
            summary_file = os.path.join(output_dir, "summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as sf:
                sf.write("=== Chat Content Statistics ===\n\n")
                sf.write(f"Total Users: {len(conversations)}\n\n")
                
                for uid, convs in sorted(conversations.items()):
                    user_info = user_map.get(uid, {'name': 'Unknown', 'email': '', 'role': ''})
                    total_msgs = sum(len(c['messages']) for c in convs.values())
                    
                    sf.write(f"User: {user_info['name']} ({user_info['email']})\n")
                    sf.write(f"User ID: {uid}\n")
                    sf.write(f"Role: {user_info['role']}\n")
                    sf.write(f"  Conversations: {len(convs)}\n")
                    sf.write(f"  Total Messages: {total_msgs}\n")
                    for conv_id, conv_data in convs.items():
                        sf.write(f"    [{conv_data['title']}]: {len(conv_data['messages'])} messages\n")
                    sf.write("\n")
            
            # Output complete CSV with user names and emails
            csv_file = os.path.join(output_dir, "pure_chats.csv")
            with open(csv_file, 'w', newline='', encoding='utf-8') as cf:
                writer = csv.DictWriter(cf, fieldnames=[
                    'user_id', 'user_name', 'user_email', 'user_role',
                    'conversation_id', 'conversation_title', 
                    'date', 'time', 'role', 'content'
                ])
                writer.writeheader()
                for uid, convs in conversations.items():
                    user_info = user_map.get(uid, {'name': 'Unknown', 'email': '', 'role': ''})
                    for conv_id, conv_data in convs.items():
                        for msg in conv_data['messages']:
                            writer.writerow({
                                'user_id': uid,
                                'user_name': user_info['name'],
                                'user_email': user_info['email'],
                                'user_role': user_info['role'],
                                'conversation_id': conv_id,
                                'conversation_title': conv_data['title'],
                                'date': msg['date'],
                                'time': msg['time'],
                                'role': msg['role'],
                                'content': msg['content']
                            })
            
            # Output separate files for each user's conversations
            for uid, convs in conversations.items():
                user_info = user_map.get(uid, {'name': 'Unknown', 'email': '', 'role': ''})
                # Folder name uses only user name (clean illegal characters)
                safe_user_name = "".join(c for c in user_info['name'] if c.isalnum() or c in (' ', '-', '_')).strip()
                user_dir = os.path.join(output_dir, safe_user_name)
                os.makedirs(user_dir, exist_ok=True)
                
                for conv_id, conv_data in convs.items():
                    # Clean illegal characters from filename
                    safe_title = "".join(c for c in conv_data['title'] if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
                    conv_file = os.path.join(user_dir, f"{safe_title}_{conv_id[:8]}.txt")
                    
                    with open(conv_file, 'w', encoding='utf-8') as cf:
                        cf.write(f"User: {user_info['name']} ({user_info['email']})\n")
                        cf.write(f"User ID: {uid}\n")
                        cf.write(f"Conversation Title: {conv_data['title']}\n")
                        cf.write(f"Conversation ID: {conv_id}\n")
                        cf.write(f"Total Messages: {len(conv_data['messages'])}\n")
                        cf.write("=" * 60 + "\n\n")
                        
                        current_date = None
                        for msg in conv_data['messages']:
                            # Add date separator when date changes
                            if msg['date'] != current_date:
                                current_date = msg['date']
                                cf.write(f"\n{'='*60}\n")
                                cf.write(f"Date: {current_date}\n")
                                cf.write(f"{'='*60}\n\n")
                            
                            cf.write(f"[{msg['time']}] [{msg['role']}]\n")
                            cf.write(f"{msg['content']}\n\n")
            
            print(f"\n✓ Analysis complete!")
            print(f"✓ Output directory: {os.path.abspath(output_dir)}")
            print(f"✓ Total users: {len(conversations)}")
            print(f"\nGrouped by user and conversation:")
            for uid, convs in sorted(conversations.items()):
                user_info = user_map.get(uid, {'name': 'Unknown', 'email': ''})
                total = sum(len(c['messages']) for c in convs.values())
                print(f"  {user_info['name']}: {len(convs)} conversations, {total} messages")
            
        else:
            print("Error: JSON top-level is not a list")

except FileNotFoundError:
    print(f"Error: File not found: {file_name}")
except json.JSONDecodeError as e:
    print(f"Error: JSON parsing error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()