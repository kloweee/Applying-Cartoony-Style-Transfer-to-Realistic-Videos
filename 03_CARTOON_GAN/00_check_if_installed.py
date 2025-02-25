import importlib.util

# For illustrative purposes.
package_name = 'tqdm'

spec = importlib.util.find_spec(package_name)
if spec is None:
    print(package_name +" is not installed")