#!/usr/bin/env python3
"""
GINsim/biolqm Integration for BNET to BND/CFG Conversion

This script demonstrates how to use GINsim/biolqm as an alternative method
for converting BNET files to BND/CFG format for MaBoSS simulations.
"""

import os

def convert_bnet_with_ginsim(bnet_file, output_bnd=None, output_cfg=None):
    """
    Convert BNET file to BND/CFG using GINsim/biolqm.
    
    This is an alternative to MaBoSS's built-in bnet_to_bnd_and_cfg function.
    
    Args:
        bnet_file (str): Path to input BNET file
        output_bnd (str): Output BND file path (optional)
        output_cfg (str): Output CFG file path (optional)
    
    Returns:
        tuple: (bnd_file, cfg_file) or (None, None) if failed
    
    Example usage:
        bnd_file, cfg_file = convert_bnet_with_ginsim("network.bnet")
    """
    try:
        import biolqm
    except ImportError:
        print("❌ biolqm not available. Install with: conda install -c colomoto biolqm")
        return None, None
    
    if not os.path.exists(bnet_file):
        print(f"❌ BNET file not found: {bnet_file}")
        return None, None
    
    try:
        print(f"🔄 Converting {os.path.basename(bnet_file)} using GINsim/biolqm...")
        
        # Load BNET file
        model = biolqm.load(bnet_file)
        
        # Convert to MaBoSS format
        maboss_model = biolqm.to_maboss(model)
        
        # Generate output filenames if not provided
        if output_bnd is None:
            base_name = os.path.splitext(bnet_file)[0]
            output_bnd = f"{base_name}_ginsim.bnd"
        
        if output_cfg is None:
            base_name = os.path.splitext(bnet_file)[0]
            output_cfg = f"{base_name}_ginsim.cfg"
        
        # Write BND file
        with open(output_bnd, "w") as f:
            maboss_model.print_bnd(out=f)
        
        # Write CFG file
        with open(output_cfg, "w") as f:
            maboss_model.print_cfg(out=f)
        
        print(f"✅ Conversion successful:")
        print(f"   BND: {output_bnd} ({os.path.getsize(output_bnd)} bytes)")
        print(f"   CFG: {output_cfg} ({os.path.getsize(output_cfg)} bytes)")
        
        return output_bnd, output_cfg
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None, None

def integrate_ginsim_into_pipeline(bnet_file):
    """
    Example of how to integrate GINsim conversion into the sensitivity analysis pipeline.
    
    This could be used as a fallback when the built-in MaBoSS conversion fails,
    or as an alternative conversion method.
    """
    print(f"🔄 Attempting GINsim conversion for {os.path.basename(bnet_file)}")
    
    # Try GINsim conversion
    bnd_file, cfg_file = convert_bnet_with_ginsim(bnet_file)
    
    if bnd_file and cfg_file:
        # Test that the files work with MaBoSS
        try:
            import maboss
            model = maboss.load(bnd_file, cfg_file)
            print("✅ GINsim-generated files compatible with MaBoSS")
            return bnd_file, cfg_file
        except Exception as e:
            print(f"⚠️  GINsim files created but MaBoSS compatibility issue: {e}")
            return None, None
    else:
        print("❌ GINsim conversion failed")
        return None, None

def demonstrate_conversion():
    """Demonstrate the GINsim conversion with available test files."""
    print("=" * 60)
    print("GINSIM/BIOLQM BNET TO BND/CFG CONVERSION DEMONSTRATION")
    print("=" * 60)
    
    # Test files to try
    test_files = [
        "example.bnet",
        "simple_test.bnet",
        "sensitivity_analysis_results/bnet_files/network_001_complete_connection_omnipath_algorithm-bfs_consensus-False_maxlen-2_only_signed-True_1.bnet"
    ]
    
    success_count = 0
    
    for bnet_file in test_files:
        if os.path.exists(bnet_file):
            print(f"\\n📁 Testing: {bnet_file}")
            print("-" * 40)
            
            bnd_file, cfg_file = convert_bnet_with_ginsim(bnet_file)
            
            if bnd_file and cfg_file:
                success_count += 1
                
                # Show file contents preview
                print("\\n📄 Generated BND file sample:")
                with open(bnd_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i < 3:
                            print(f"   {line.rstrip()}")
                        else:
                            print("   ...")
                            break
                
                print("\\n📄 Generated CFG file sample:")
                with open(cfg_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i < 3:
                            print(f"   {line.rstrip()}")
                        else:
                            print("   ...")
                            break
                
                # Clean up demo files
                try:
                    os.remove(bnd_file)
                    os.remove(cfg_file)
                    print("\\n🧹 Demo files cleaned up")
                except:
                    pass
            else:
                print("❌ Conversion failed")
    
    print(f"\\n{'='*60}")
    print(f"SUMMARY: {success_count}/{len([f for f in test_files if os.path.exists(f)])} conversions successful")
    
    if success_count > 0:
        print("\\n🎉 GINsim/biolqm conversion is working!")
        print("\\nKey Benefits:")
        print("✓ Alternative conversion engine")
        print("✓ May handle certain BNET formats better") 
        print("✓ Can serve as fallback when built-in conversion fails")
        print("✓ Part of the CoLoMoTo ecosystem for biological modeling")
        
        print("\\nUsage in pipeline:")
        print("1. Try built-in MaBoSS conversion first")
        print("2. If that fails, use GINsim as backup")
        print("3. Both produce compatible BND/CFG files")

if __name__ == "__main__":
    demonstrate_conversion()