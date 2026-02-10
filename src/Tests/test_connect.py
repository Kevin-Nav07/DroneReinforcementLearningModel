import cflib.crtp
import logging

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

def find_uri():
    """
    Initializes drivers and scans for available Crazyflies.
    """
    ###Initialize the low-level drivers (required for scanning)
    cflib.crtp.init_drivers()

    print("Scanning interfaces for Crazyflies... Please wait.")
    
    ###Scan for available interfaces
    available = cflib.crtp.scan_interfaces()

    ###Process results
    if len(available) == 0:
        print("No Crazyflies found. Check power and radio connection.")
    else:
        print("Crazyflies found:")
        for interface in available:
            ### interface[0] contains the URI string
            print(f" - {interface[0]}, interface: ")
            
### Main execution
if __name__ == '__main__':
    find_uri()