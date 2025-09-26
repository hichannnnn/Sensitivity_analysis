#!/usr/bin/env python3
"""
GINsim Priority-Based Converter
Creates BND/CFG files using the most processed BNET files available
Priority: sanitized_deduplicated > sanitized > original
"""

import os
import glob
import biolqm

def find_best_bnet_file(maboss_dir, network_num):
    """Find the best (most processed) BNET file for a given network number"""
    
    # Find all files matching the network number pattern
    pattern = os.path.join(maboss_dir, f"network_{network_num:03d}_*.bnet")
    all_files = glob.glob(pattern)
    
    if not all_files:
        return None, "original"
    
    # Sort by processing level priority
    sanitized_deduplicated = [f for f in all_files if f.endswith('_sanitized_deduplicated.bnet')]
    sanitized_only = [f for f in all_files if f.endswith('_sanitized.bnet') and not f.endswith('_deduplicated.bnet')]
    original_only = [f for f in all_files if not ('_sanitized' in f or '_deduplicated' in f)]
    
    # Return the best available option
    if sanitized_deduplicated:
        return sanitized_deduplicated[0], "sanitized_deduplicated"
    elif sanitized_only:
        return sanitized_only[0], "sanitized"
    elif original_only:
        return original_only[0], "original"
    else:
        return None, "none"

def convert_with_ginsim(bnet_file, output_dir, network_info):
    """Convert BNET to BND/CFG using GINsim"""
    
    try:
        print(f"🔄 Converting: {os.path.basename(bnet_file)}")
        print(f"   Type: {network_info['type']}")
        
        # Load the BNET model
        model = biolqm.load(bnet_file)
        
        # Generate output names
        base_name = os.path.splitext(os.path.basename(bnet_file))[0]
        bnd_file = os.path.join(output_dir, f"{base_name}_ginsim.bnd")
        cfg_file = os.path.join(output_dir, f"{base_name}_ginsim.cfg")
        
        # Save as MaBoSS format
        biolqm.save(model, bnd_file, "bnd")
        biolqm.save(model, cfg_file, "cfg")
        
        # Verify files were created and get sizes
        if os.path.exists(bnd_file) and os.path.exists(cfg_file):
            bnd_size = os.path.getsize(bnd_file)
            cfg_size = os.path.getsize(cfg_file)
            
            print(f"   ✅ BND: {os.path.basename(bnd_file)} ({bnd_size} bytes)")
            print(f"   ✅ CFG: {os.path.basename(cfg_file)} ({cfg_size} bytes)")
            
            return True, {
                'bnd_file': bnd_file,
                'cfg_file': cfg_file,
                'bnd_size': bnd_size,
                'cfg_size': cfg_size
            }
        else:
            print(f"   ❌ Files not created")
            return False, "Files not created"
            
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return False, str(e)

def main():
    """Main conversion function with priority-based file selection"""
    
    print("🚀 GINsim Priority-Based Converter")
    print("=" * 60)
    print("Priority: sanitized_deduplicated > sanitized > original")
    print("=" * 60)
    
    # Setup directories
    workspace_dir = "/home/ishimizu/Desktop/github/Materials_for_Presentation"
    maboss_dir = os.path.join(workspace_dir, "sensitivity_analysis_results", "maboss_results")
    output_dir = os.path.join(workspace_dir, "sensitivity_analysis_results", "ginsim_results")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Target networks (excluding 008, 024)
    target_networks = (
        list(range(1, 8)) +    # 001-007
        list(range(9, 24)) +   # 009-023  
        list(range(25, 29))    # 025-028
    )
    
    print(f"📁 Source: {os.path.relpath(maboss_dir, workspace_dir)}")
    print(f"📁 Output: {os.path.relpath(output_dir, workspace_dir)}")
    print(f"🎯 Networks: {len(target_networks)} (excluding 008, 024)")
    print()
    
    results = []
    successful_count = 0
    
    for network_num in target_networks:
        print(f"[{network_num:03d}] Processing network {network_num:03d}...")
        
        # Find best BNET file for this network
        best_file, file_type = find_best_bnet_file(maboss_dir, network_num)
        
        if not best_file:
            print(f"   ❌ No BNET file found")
            continue
        
        # Convert with GINsim
        network_info = {'type': file_type, 'network': network_num}
        success, output = convert_with_ginsim(best_file, output_dir, network_info)
        
        if success:
            successful_count += 1
        
        print()
    
    # Final report
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"📊 Successful: {successful_count}/{len(target_networks)}")
    
    # Count files
    bnd_count = len([f for f in os.listdir(output_dir) if f.endswith('.bnd')])
    cfg_count = len([f for f in os.listdir(output_dir) if f.endswith('.cfg')])
    
    print(f"📁 Created: {bnd_count} BND + {cfg_count} CFG = {bnd_count + cfg_count} files")
    print(f"🎉 Ready for MaBoSS simulations!")

if __name__ == "__main__":
    main()