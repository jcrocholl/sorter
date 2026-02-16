from cluster_images import cluster_images, main


def test_cluster_images_basic(tmp_path):
    # Mock files
    (tmp_path / "20230523_105352000_a.jpg").touch()
    (tmp_path / "20230523_105352500_b.jpg").touch()  # 0.5s gap
    (tmp_path / "20230523_105354000_c.jpg").touch()  # 1.5s gap

    # default threshold is 0.5s.
    # a and b are 0.5s apart, so they should be clustered if threshold >= 0.5
    # c is 1.5s after b, so it should be in a new cluster

    clusters = cluster_images(tmp_path, gap_threshold_seconds=0.5)

    assert len(clusters) == 2
    assert len(clusters[0]) == 2
    assert clusters[0][0][1] == "20230523_105352000_a.jpg"
    assert clusters[0][1][1] == "20230523_105352500_b.jpg"
    assert len(clusters[1]) == 1
    assert clusters[1][0][1] == "20230523_105354000_c.jpg"


def test_cluster_images_no_files(tmp_path):
    clusters = cluster_images(tmp_path)
    assert clusters == []


def test_cluster_images_invalid_filenames(tmp_path):
    (tmp_path / "invalid_name.jpg").touch()
    (tmp_path / "20230523_105352000_a.jpg").touch()

    clusters = cluster_images(tmp_path)
    assert len(clusters) == 1
    assert len(clusters[0]) == 1
    assert clusters[0][0][1] == "20230523_105352000_a.jpg"


def test_main_with_arguments(tmp_path, capsys):
    (tmp_path / "20230523_105352000_a.jpg").touch()

    # Test main with the tmp_path as an argument
    main(["cluster_images.py", str(tmp_path)])

    captured = capsys.readouterr()
    assert "Total Clusters: 1" in captured.out
    assert "Total Images:   1" in captured.out
    assert "Train:   1 clusters (100.0%),    1 images" in captured.out


def test_main_no_images(tmp_path, capsys):
    # Test main on an empty directory
    main(["cluster_images.py", str(tmp_path)])

    captured = capsys.readouterr()
    assert "No images found to cluster." in captured.out
