








# Use packs to sweep over different resources 

# Use job arrays when sweeping over different python script parameters



# Requirements: 
- Needs to be general in the sense that we could use an arbitrary template job
- we use .yaml files for defining configs. .yaml file is broken into 'groups' (top level keys)
- each group defines: 
  - Resources (controlled through packs)
  - Script arguments (controlled through job arrays)
  - template job script that is used 



