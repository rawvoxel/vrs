//! What the stress test runs, and what it expects of each run.

use crate::camera;
use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Debug)]
pub enum Camera {
    Identity,
    /// Scales by two, so the wall fills the screen.
    Zoom,
    Perspective,
}

impl Camera {
    pub fn view_proj(self, aspect: f32) -> (Mat4, Mat4) {
        match self {
            Camera::Identity => (Mat4::IDENTITY, Mat4::IDENTITY),
            Camera::Zoom => (Mat4::IDENTITY, Mat4::from_scale(Vec3::new(2.0, 2.0, 1.0))),
            Camera::Perspective => {
                let view =
                    Mat4::look_at_rh(Vec3::new(0.0, 0.0, -1.5), Vec3::new(0.0, 0.0, 0.5), Vec3::Y);
                (view, camera::perspective(60.0, aspect, 0.1, 10.0))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    /// Draw counts must match the meshes in view exactly.
    FrustumOnly,
    /// At least every mesh the wall can't hide, at most every mesh in view.
    Occluded,
    /// Only the wall, which hides everything else.
    OnlyWall,
}

/// Forcing the mesh pool to repack, after the scene is filled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolTest {
    None,
    /// Every other chunk is destroyed, then a mesh too big for any gap but not for all of
    /// them is created, so the pool must defragment.
    Defragment,
    /// A mesh bigger than all the free space is created, so the pool must grow.
    Grow,
}

pub struct Scenario {
    pub name: &'static str,
    pub frames: u32,
    pub camera: Camera,
    pub wall: bool,
    /// Backface culling must skip the wall, so it hides nothing.
    pub wall_faces_away: bool,
    pub occlusion: bool,
    /// Remesh chunks every frame, and resize and minimize the window.
    pub remesh: bool,
    /// The pool only grows for `PoolTest::Grow`.
    pub pool: PoolTest,
    pub expect: Expect,
}

pub const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "frustum culling",
        frames: 60,
        camera: Camera::Identity,
        wall: false,
        wall_faces_away: false,
        occlusion: false,
        remesh: false,
        pool: PoolTest::None,
        expect: Expect::FrustumOnly,
    },
    Scenario {
        name: "frustum culling, zoomed in",
        frames: 60,
        camera: Camera::Zoom,
        wall: false,
        wall_faces_away: false,
        occlusion: false,
        remesh: false,
        pool: PoolTest::None,
        expect: Expect::FrustumOnly,
    },
    Scenario {
        name: "occlusion culling",
        frames: 60,
        camera: Camera::Identity,
        wall: true,
        wall_faces_away: false,
        occlusion: true,
        remesh: false,
        pool: PoolTest::None,
        expect: Expect::Occluded,
    },
    Scenario {
        name: "occlusion culling, perspective",
        frames: 60,
        camera: Camera::Perspective,
        wall: true,
        wall_faces_away: false,
        occlusion: true,
        remesh: false,
        pool: PoolTest::None,
        expect: Expect::Occluded,
    },
    Scenario {
        name: "backface culling",
        frames: 60,
        camera: Camera::Perspective,
        wall: true,
        wall_faces_away: true,
        occlusion: true,
        remesh: false,
        pool: PoolTest::None,
        expect: Expect::Occluded,
    },
    Scenario {
        name: "occlusion culling, wall fills the screen",
        frames: 60,
        camera: Camera::Zoom,
        wall: true,
        wall_faces_away: false,
        occlusion: true,
        remesh: false,
        pool: PoolTest::None,
        expect: Expect::OnlyWall,
    },
    Scenario {
        name: "remeshing",
        frames: 600,
        camera: Camera::Perspective,
        wall: true,
        wall_faces_away: false,
        occlusion: true,
        remesh: true,
        pool: PoolTest::None,
        expect: Expect::Occluded,
    },
    Scenario {
        name: "defragmenting the mesh pool",
        frames: 60,
        camera: Camera::Perspective,
        wall: true,
        wall_faces_away: false,
        occlusion: true,
        remesh: false,
        pool: PoolTest::Defragment,
        expect: Expect::Occluded,
    },
    Scenario {
        name: "growing the mesh pool",
        frames: 60,
        camera: Camera::Perspective,
        wall: true,
        wall_faces_away: false,
        occlusion: true,
        remesh: false,
        pool: PoolTest::Grow,
        expect: Expect::Occluded,
    },
];
