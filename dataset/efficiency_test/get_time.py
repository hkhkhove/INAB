import os


def get_inab(prot):
    with open(
        f"/home/hzeng/p/INAB/dataset/efficiency_test/features/{prot}_time.txt", "r"
    ) as f:
        lines = f.readlines()

    total_time = lines[2].split(":")[1].strip()[:-1]
    total_time = float(total_time)

    return total_time


def get_nabind(prot):
    with open(
        f"/home2/hzeng/data/NABind_efficiency_test/features/{prot}/time.txt", "r"
    ) as f:
        lines = f.readlines()

    total_time = lines[2].split(":")[1].strip()
    total_time = float(total_time)

    return total_time


def main():
    with open(
        "/home/hzeng/p/INAB/dataset/efficiency_test/length_summary.txt", "r"
    ) as f:
        lines = f.readlines()
    prot_100 = [
        e.split()[0].strip() for e in lines[:10] if int(e.split()[1]) - 100 <= 10
    ]
    prot_300 = [
        e.split()[0].strip() for e in lines[10:20] if int(e.split()[1]) - 300 <= 10
    ]
    prot_500 = [
        e.split()[0].strip() for e in lines[20:30] if int(e.split()[1]) - 500 <= 10
    ]
    prot_1000 = [
        e.split()[0].strip() for e in lines[30:40] if int(e.split()[1]) - 1000 <= 10
    ]
    prot_gt_1500 = [
        e.split()[0].strip() for e in lines[40:] if int(e.split()[1]) >= 1500
    ]

    prots_list = [prot_100, prot_300, prot_500, prot_1000, prot_gt_1500]

    inab_times = [[] for _ in range(len(prots_list))]
    nabind_times = [[] for _ in range(len(prots_list))]
    for i, prots in enumerate(prots_list):
        for prot in prots:
            inab_time = get_inab(prot)
            inab_times[i].append(inab_time)
            try:
                nabind_time = get_nabind(prot)
                nabind_times[i].append(nabind_time)
            except Exception as e:
                print(f"Error processing {prot}: {e}")

    print("INAB times:")
    for i, times in enumerate(inab_times):
        print(times)
        print(f"Average time for {len(times)} proteins: {sum(times) / len(times):.2f}s")

    print("NABind times:")
    for i, times in enumerate(nabind_times):
        print(times)
        print(f"Average time for {len(times)} proteins: {sum(times) / len(times):.2f}s")


main()
