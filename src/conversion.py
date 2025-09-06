

import argparse, hashlib, sys
from scapy.all import PcapReader, PcapWriter, Ether, IPv6, UDP, TCP
from scapy.config import conf


conf.dot15d4_protocol = "sixlowpan"


SixLoWPAN = SixLoWPAN_IPHC = sixlowpan_decompress = sixlowpan_set_context = None
try:
    from scapy.layers.sixlowpan import (
        SixLoWPAN, SixLoWPAN_IPHC, sixlowpan_decompress, sixlowpan_set_context
    )
except Exception:
    try:
        from scapy.contrib.sixlowpan import (
            SixLoWPAN, SixLoWPAN_IPHC, sixlowpan_decompress, sixlowpan_set_context
        )
    except Exception:
        pass


try:
    from scapy.layers.dot15d4 import Dot15d4, Dot15d4Data, Dot15d4FCS
except Exception:
    Dot15d4 = Dot15d4Data = Dot15d4FCS = None

def mac_from_ieee(src64=None, src16=None):
    if src64 is not None:
        b = int(src64).to_bytes(8, "big")
        return ":".join(f"{x:02x}" for x in b[2:])
    if src16 is not None:
        h = hashlib.md5(int(src16).to_bytes(2, "big")).digest()[:6]
        h = bytes([h[0] | 0x02]) + h[1:]
        return ":".join(f"{x:02x}" for x in h)
    return "02:00:00:00:00:00"

def ieee_addrs(pkt):

    src64 = dst64 = src16 = dst16 = None
    d = pkt
    while d:
        nm = getattr(d, "name", "").lower()
        if "802.15.4" in nm:
            for attr in ("src_addr", "dest_addr"):
                if hasattr(d, attr):
                    val = getattr(d, attr)
                    if attr == "src_addr":
                        (src64, src16) = (val, None) if val and val > 0xFFFF else (None, val)
                    else:
                        (dst64, dst16) = (val, None) if val and val > 0xFFFF else (None, val)
            break
        d = d.payload
    return src64, src16, dst64, dst16

def iid_from_ll(src64=None, src16=None):

    if src64 is not None:
        b = int(src64).to_bytes(8, "big")

        b = bytes([b[0] ^ 0x02]) + b[1:]

        return ":".join(f"{int.from_bytes(b[i:i+2],'big'):x}" for i in range(0,8,2))
    if src16 is not None:

        x = int(src16)
        return "0:ff:fe00:{:x}".format(x)
    return None

def fix_iid(ip6, ll_src64, ll_src16, ll_dst64, ll_dst16, prefix="fe80::"):
    """If ip6.src/dst lost IID (== prefix or '::'), rebuild it."""
    from ipaddress import IPv6Address, IPv6Network

    pfx = IPv6Network(prefix + "/64", strict=False)

    def _set(addr_str, iid_str):
        if iid_str is None:
            return addr_str

        iid_parts = iid_str.split(":")

        full = (pfx.network_address.exploded.split(":")[:4] + iid_parts)

        addr = IPv6Address(":".join(full)).compressed
        return addr

    if ip6.src in ("::",) or ip6.src.startswith(prefix) and ip6.src == prefix:
        iid = iid_from_ll(ll_src64, ll_src16)
        if iid:
            ip6.src = _set(prefix, iid)
    if ip6.dst in ("::",) or ip6.dst.startswith(prefix) and ip6.dst == prefix:
        iid = iid_from_ll(ll_dst64, ll_dst16)
        if iid:
            ip6.dst = _set(prefix, iid)

def as_ipv6(pkt):
    if IPv6 in pkt:
        return pkt[IPv6]
    if SixLoWPAN and (pkt.haslayer(SixLoWPAN_IPHC) or pkt.haslayer(SixLoWPAN)):
        s64, s16, d64, d16 = ieee_addrs(pkt)
        try:
            return sixlowpan_decompress(pkt, src_ll=s64 or s16, dst_ll=d64 or d16)
        except Exception:
            return None
    return None

def convert(infile, outfile, keep_all=False, prefix="fe80::", cid=0):

    if sixlowpan_set_context:
        try:
            sixlowpan_set_context(int(cid), prefix, 64)
        except Exception:
            pass

    r = PcapReader(infile)
    w = PcapWriter(outfile, linktype=1, sync=True)
    n_in = n_out = 0

    for pkt in r:
        n_in += 1
        ip6 = as_ipv6(pkt)
        if ip6 is None:
            continue


        s64, s16, d64, d16 = ieee_addrs(pkt)
        fix_iid(ip6, s64, s16, d64, d16, prefix=prefix)

        if not keep_all and not (UDP in ip6 or TCP in ip6):
            continue

        eth = Ether(
            src=mac_from_ieee(s64, s16),
            dst=mac_from_ieee(d64, d16),
            type=0x86DD
        ) / ip6
        if UDP in eth: eth[UDP].chksum = None
        if TCP in eth: eth[TCP].chksum = None
        eth[IPv6].plen = None
        try: eth.time = pkt.time
        except Exception: pass

        w.write(eth); n_out += 1

    r.close(); w.close()
    print(f"[+] Processed {n_in} packets; wrote {n_out} IPv6 Ethernet frames to {outfile}")

def main():
    ap = argparse.ArgumentParser(description="6LoWPAN->Ethernet IPv6 converter")
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--prefix", default="fe80::", help="6LoWPAN context prefix (default fe80::)")
    ap.add_argument("--cid", default=0, type=int, help="6LoWPAN context ID (default 0)")
    ap.add_argument("--keep-all", action="store_true", help="Keep ICMPv6/others (not only UDP/TCP)")
    args = ap.parse_args()
    convert(args.infile, args.outfile, keep_all=args.keep_all, prefix=args.prefix, cid=args.cid)

if __name__ == "__main__":
    if SixLoWPAN is None:
        print("[-] Scapy 6LoWPAN layer not found. Install/upgrade python3-scapy.")
        sys.exit(1)
    main()
