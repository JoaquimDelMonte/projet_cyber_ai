import pyshark
import numpy as np
import time
import csv
from collections import defaultdict

# Dictionnaire global pour stocker les flux
flows = {}

def get_flow_key(packet):
    try:
        ip_layer = packet.ip
        proto = packet.transport_layer  # "TCP" ou "UDP"
        src = ip_layer.src
        dst = ip_layer.dst
        if proto == "TCP":
            sport = packet.tcp.srcport
            dport = packet.tcp.dstport
        elif proto == "UDP":
            sport = packet.udp.srcport
            dport = packet.udp.dstport
        else:
            sport, dport = None, None
        return (src, dst, sport, dport, proto)
    except AttributeError:
        return None

def update_flow(flow, packet, timestamp, direction):
    pkt_len = int(packet.length)
    flow['all_timestamps'].append(timestamp)
    flow['packet_lengths'].append(pkt_len)
    if direction == "fwd":
        flow['fwd_packet_lengths'].append(pkt_len)
        flow['fwd_timestamps'].append(timestamp)
    elif direction == "bwd":
        flow['bwd_packet_lengths'].append(pkt_len)
        flow['bwd_timestamps'].append(timestamp)
    
    # Comptage des flags TCP (si TCP)
    if hasattr(packet, 'tcp'):
        for flag in ['fin', 'syn', 'rst', 'psh', 'ack', 'urg']:
            try:
                value = int(getattr(packet.tcp, flag + '_flag'))
            except AttributeError:
                value = 0
            flow['tcp_flags'][flag] += value
            if flag == 'psh':
                if direction == "fwd":
                    flow['fwd_psh_flags'] += value
                else:
                    flow['bwd_psh_flags'] += value
            if flag == 'urg':
                if direction == "fwd":
                    flow['fwd_urg_flags'] += value
                else:
                    flow['bwd_urg_flags'] += value
        # Longueur d'en-tête TCP (si disponible)
        try:
            hdr_len = int(packet.tcp.hdr_len)
        except AttributeError:
            hdr_len = 0
        if direction == "fwd":
            flow['fwd_header_lengths'].append(hdr_len)
        else:
            flow['bwd_header_lengths'].append(hdr_len)

def process_packet(packet):
    timestamp = float(packet.sniff_timestamp)
    key = get_flow_key(packet)
    if key is None:
        return
    # Initialiser le flux s'il n'existe pas encore
    if key not in flows:
        flows[key] = {
            'src': key[0],
            'dst': key[1],
            'sport': key[2],
            'dport': key[3],
            'protocol': key[4],
            'start_time': timestamp,
            'end_time': timestamp,
            'all_timestamps': [timestamp],
            'packet_lengths': [int(packet.length)],
            'fwd_packet_lengths': [],
            'bwd_packet_lengths': [],
            'fwd_timestamps': [],
            'bwd_timestamps': [],
            'tcp_flags': defaultdict(int),
            'fwd_psh_flags': 0,
            'bwd_psh_flags': 0,
            'fwd_urg_flags': 0,
            'bwd_urg_flags': 0,
            'fwd_header_lengths': [],
            'bwd_header_lengths': [],
        }
        # Premier paquet considéré comme "forward"
        flows[key]['fwd_packet_lengths'].append(int(packet.length))
        flows[key]['fwd_timestamps'].append(timestamp)
    else:
        flow = flows[key]
        flow['end_time'] = timestamp
        # On considère "fwd" si l'IP source correspond à celle du premier paquet
        direction = "fwd" if packet.ip.src == flow['src'] else "bwd"
        update_flow(flow, packet, timestamp, direction)

def compute_metrics(flow):
    metrics = {}
    duration = flow['end_time'] - flow['start_time'] if flow['end_time'] > flow['start_time'] else 1e-6
    total_packets = len(flow['packet_lengths'])
    total_bytes = sum(flow['packet_lengths'])
    
    # Informations de base
    metrics['Source IP'] = flow['src']
    metrics['Destination Port'] = flow['dport']
    metrics['Flow Duration'] = duration
    metrics['Total Fwd Packets'] = len(flow['fwd_packet_lengths'])
    metrics['Total Backward Packets'] = len(flow['bwd_packet_lengths'])
    metrics['Total Length of Fwd Packets'] = sum(flow['fwd_packet_lengths'])
    metrics['Total Length of Bwd Packets'] = sum(flow['bwd_packet_lengths'])
    
    # Statistiques sur les longueurs de paquets (forward)
    if flow['fwd_packet_lengths']:
        metrics['Fwd Packet Length Max'] = max(flow['fwd_packet_lengths'])
        metrics['Fwd Packet Length Min'] = min(flow['fwd_packet_lengths'])
        metrics['Fwd Packet Length Mean'] = np.mean(flow['fwd_packet_lengths'])
        metrics['Fwd Packet Length Std'] = np.std(flow['fwd_packet_lengths'])
    else:
        metrics['Fwd Packet Length Max'] = metrics['Fwd Packet Length Min'] = metrics['Fwd Packet Length Mean'] = metrics['Fwd Packet Length Std'] = 0
        
    # Statistiques sur les longueurs de paquets (backward)
    if flow['bwd_packet_lengths']:
        metrics['Bwd Packet Length Max'] = max(flow['bwd_packet_lengths'])
        metrics['Bwd Packet Length Min'] = min(flow['bwd_packet_lengths'])
        metrics['Bwd Packet Length Mean'] = np.mean(flow['bwd_packet_lengths'])
        metrics['Bwd Packet Length Std'] = np.std(flow['bwd_packet_lengths'])
    else:
        metrics['Bwd Packet Length Max'] = metrics['Bwd Packet Length Min'] = metrics['Bwd Packet Length Mean'] = metrics['Bwd Packet Length Std'] = 0
        
    metrics['Flow Bytes/s'] = total_bytes / duration
    metrics['Flow Packets/s'] = total_packets / duration

    # Inter-arrival times globales
    all_times = sorted(flow['all_timestamps'])
    if len(all_times) > 1:
        iats = np.diff(all_times)
        metrics['Flow IAT Mean'] = np.mean(iats)
        metrics['Flow IAT Std'] = np.std(iats)
        metrics['Flow IAT Max'] = np.max(iats)
        metrics['Flow IAT Min'] = np.min(iats)
    else:
        metrics['Flow IAT Mean'] = metrics['Flow IAT Std'] = metrics['Flow IAT Max'] = metrics['Flow IAT Min'] = 0

    # IAT pour forward
    if len(flow['fwd_timestamps']) > 1:
        fwd_iats = np.diff(sorted(flow['fwd_timestamps']))
        metrics['Fwd IAT Total'] = np.sum(fwd_iats)
        metrics['Fwd IAT Mean'] = np.mean(fwd_iats)
        metrics['Fwd IAT Std'] = np.std(fwd_iats)
        metrics['Fwd IAT Max'] = np.max(fwd_iats)
        metrics['Fwd IAT Min'] = np.min(fwd_iats)
    else:
        metrics['Fwd IAT Total'] = metrics['Fwd IAT Mean'] = metrics['Fwd IAT Std'] = metrics['Fwd IAT Max'] = metrics['Fwd IAT Min'] = 0

    # IAT pour backward
    if len(flow['bwd_timestamps']) > 1:
        bwd_iats = np.diff(sorted(flow['bwd_timestamps']))
        metrics['Bwd IAT Total'] = np.sum(bwd_iats)
        metrics['Bwd IAT Mean'] = np.mean(bwd_iats)
        metrics['Bwd IAT Std'] = np.std(bwd_iats)
        metrics['Bwd IAT Max'] = np.max(bwd_iats)
        metrics['Bwd IAT Min'] = np.min(bwd_iats)
    else:
        metrics['Bwd IAT Total'] = metrics['Bwd IAT Mean'] = metrics['Bwd IAT Std'] = metrics['Bwd IAT Max'] = metrics['Bwd IAT Min'] = 0

    # Flags PSH et URG par direction
    metrics['Fwd PSH Flags'] = flow['fwd_psh_flags']
    metrics['Bwd PSH Flags'] = flow['bwd_psh_flags']
    metrics['Fwd URG Flags'] = flow['fwd_urg_flags']
    metrics['Bwd URG Flags'] = flow['bwd_urg_flags']

    # Moyenne des longueurs d'en-tête (si disponibles)
    metrics['Fwd Header Length'] = np.mean(flow['fwd_header_lengths']) if flow['fwd_header_lengths'] else 0
    metrics['Bwd Header Length'] = np.mean(flow['bwd_header_lengths']) if flow['bwd_header_lengths'] else 0

    # Débit par direction (packets/s)
    metrics['Fwd Packets/s'] = len(flow['fwd_packet_lengths']) / duration
    metrics['Bwd Packets/s'] = len(flow['bwd_packet_lengths']) / duration

    # Statistiques globales sur les longueurs de paquets
    if flow['packet_lengths']:
        metrics['Min Packet Length'] = min(flow['packet_lengths'])
        metrics['Max Packet Length'] = max(flow['packet_lengths'])
        metrics['Packet Length Mean'] = np.mean(flow['packet_lengths'])
        metrics['Packet Length Std'] = np.std(flow['packet_lengths'])
        metrics['Packet Length Variance'] = np.var(flow['packet_lengths'])
        metrics['Average Packet Size'] = np.mean(flow['packet_lengths'])
    else:
        metrics['Min Packet Length'] = metrics['Max Packet Length'] = metrics['Packet Length Mean'] = metrics['Packet Length Std'] = metrics['Packet Length Variance'] = metrics['Average Packet Size'] = 0

    # Comptage global des flags TCP
    metrics['FIN Flag Count'] = flow['tcp_flags'].get('fin', 0)
    metrics['SYN Flag Count'] = flow['tcp_flags'].get('syn', 0)
    metrics['RST Flag Count'] = flow['tcp_flags'].get('rst', 0)
    metrics['PSH Flag Count'] = flow['tcp_flags'].get('psh', 0)
    metrics['ACK Flag Count'] = flow['tcp_flags'].get('ack', 0)
    metrics['URG Flag Count'] = flow['tcp_flags'].get('urg', 0)
    # Placeholders pour CWE et ECE
    metrics['CWE Flag Count'] = 0
    metrics['ECE Flag Count'] = 0

    # Ratio Down/Up (backward/forward)
    fwd_count = len(flow['fwd_packet_lengths'])
    bwd_count = len(flow['bwd_packet_lengths'])
    metrics['Down/Up Ratio'] = bwd_count / fwd_count if fwd_count > 0 else 0

    # Moyennes de segments par direction
    metrics['Avg Fwd Segment Size'] = np.mean(flow['fwd_packet_lengths']) if flow['fwd_packet_lengths'] else 0
    metrics['Avg Bwd Segment Size'] = np.mean(flow['bwd_packet_lengths']) if flow['bwd_packet_lengths'] else 0

    # Placeholders pour les métriques bulk
    metrics['Fwd Avg Bytes/Bulk'] = 0
    metrics['Fwd Avg Packets/Bulk'] = 0
    metrics['Fwd Avg Bulk Rate'] = 0
    metrics['Bwd Avg Bytes/Bulk'] = 0
    metrics['Bwd Avg Packets/Bulk'] = 0
    metrics['Bwd Avg Bulk Rate'] = 0

    # Placeholders pour les sous-flux
    metrics['Subflow Fwd Packets'] = 0
    metrics['Subflow Fwd Bytes'] = 0
    metrics['Subflow Bwd Packets'] = 0
    metrics['Subflow Bwd Bytes'] = 0

    # Placeholders pour les tailles de fenêtre initiale
    metrics['Init_Win_bytes_forward'] = 0
    metrics['Init_Win_bytes_backward'] = 0

    # Placeholders pour act_data_pkt_fwd et min_seg_size_forward
    metrics['act_data_pkt_fwd'] = 0
    metrics['min_seg_size_forward'] = 0

    # Placeholders pour les mesures d'activité/inaction
    metrics['Active Mean'] = 0
    metrics['Active Std'] = 0
    metrics['Active Max'] = 0
    metrics['Active Min'] = 0
    metrics['Idle Mean'] = 0
    metrics['Idle Std'] = 0
    metrics['Idle Max'] = 0
    metrics['Idle Min'] = 0

    # Label (à renseigner si besoin)
    metrics['Label'] = 'Unknown'
    
    return metrics

def main():
    # Remplacer 'Wi-Fi' par le nom de l'interface de capture adaptée
    capture = pyshark.LiveCapture(interface='Wi-Fi')
    print("Capture en direct démarrée sur l'interface Wi-Fi...")

    # Liste des champs (ordre exact pour le CSV)
    fieldnames = [
        "Source IP", "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
        "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
        "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
        "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
        "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
        "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
        "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
        "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
        "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
        "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max",
        "Idle Min", "Label"
    ]

    # Ouverture du fichier CSV pour écriture
    with open("flows.csv", mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Boucle de capture en continu
        for packet in capture.sniff_continuously():
            process_packet(packet)
            now = time.time()
            keys_to_remove = []
            # Vérifier les flux inactifs (plus de 10 secondes sans nouveau paquet)
            for key, flow in flows.items():
                if now - flow['end_time'] > 10:
                    stats = compute_metrics(flow)
                    writer.writerow(stats)
                    csvfile.flush()
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del flows[key]

if __name__ == '__main__':
    main()
