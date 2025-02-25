async def main(indir):
    # Scan dirs for files and populate a list
    filepaths = []
    for path, dirs, files in os.walk(indir):
        for basename in files:
            filepath = os.path.join(path, basename)
            filepaths.append(filepath)

    """Create the "process pool" of 4 and run asyncio.
    The processes will execute the worker function
    concurrently with each file path as parameter
    """
    loop = asyncio.get_running_loop()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [loop.run_in_executor(executor, func, f) for f in filepaths]
        for fut in asyncio.as_completed(futures):
            try:
                filepath = await fut
                print(filepath)
            except Exception as exc:
                print("failed one job")
