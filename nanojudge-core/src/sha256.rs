const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const H_INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn ch(x: u32, y: u32, z: u32) -> u32 { (x & y) ^ (!x & z) }
fn maj(x: u32, y: u32, z: u32) -> u32 { (x & y) ^ (x & z) ^ (y & z) }
fn big_sigma0(x: u32) -> u32 { x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22) }
fn big_sigma1(x: u32) -> u32 { x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25) }
fn small_sigma0(x: u32) -> u32 { x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3) }
fn small_sigma1(x: u32) -> u32 { x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10) }

pub fn sha256(input: &[u8]) -> [u8; 32] {
    let bit_len = (input.len() as u64) * 8;

    // Pad: append 0x80, then zeros, then 8-byte big-endian bit length,
    // so total length is a multiple of 64 bytes (512 bits).
    let mut msg = Vec::with_capacity(input.len() + 72);
    msg.extend_from_slice(input);
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    let mut h = H_INIT;

    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            w[i] = small_sigma1(w[i - 2])
                .wrapping_add(w[i - 7])
                .wrapping_add(small_sigma0(w[i - 15]))
                .wrapping_add(w[i - 16]);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let t1 = hh
                .wrapping_add(big_sigma1(e))
                .wrapping_add(ch(e, f, g))
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::sha256;

    #[test]
    fn empty() {
        let digest = sha256(b"");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn abc() {
        let digest = sha256(b"abc");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn two_block() {
        let input = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let digest = sha256(input);
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    }
}
